import os
import time
from argparse import ArgumentParser

import scipy.io as sio
# from skimage.measure import compare_ssim as ssim
# from utils import imread_CS_py, img2col_py, col2im_CS_py, psnr, add_test_noise, write_data,get_cond
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ISTA_Netpp

parser = ArgumentParser(description='ISTA-Net-plus-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='3', help='gpu index')
# parser.add_argument('--data_dir', type=str, default='cs_train400_png', help='training data directory')
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--patch_size', type=int, default=33, help='from {1, 4, 10, 25, 40, 50}')

# parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
# parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
# parser.add_argument('--data_dir_org', type=str, default='data', help='training data directory')
# parser.add_argument('--log_dir', type=str, default='log', help='log directory')
# parser.add_argument('--ext', type=str, default='.png', help='training data directory')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix_single', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model_single', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='my_data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')

args = parser.parse_args()

localtime = time.localtime(time.time())
nowtime = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(
    localtime.tm_hour) + '_' + str(localtime.tm_min) + '_' + str(localtime.tm_sec)

tensorBoardPath = './singlePixel_ISTANetpp/%s' % nowtime
if not os.path.exists(tensorBoardPath):
    os.makedirs(tensorBoardPath)
writer = SummaryWriter('%s/' % tensorBoardPath)

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list

torch.cuda.set_device(int(gpu_list))

y_size = 5000
gamma = torch.Tensor([[y_size / 16384]]).cuda()

n_input = 5000
n_output = 16384
nrtrain = 780
batch_size = 32

Phi_data_Name = './%s/A.mat' % args.matrix_dir  # 8000/16384
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['A']

Training_data_Name = 'X0.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['X']

Y_data_Name = 'Y0.mat'
Y_data = sio.loadmat('./%s/%s' % (args.data_dir, Y_data_Name))
Y_data = Y_data['Y']

Qinit_Name = './%s/Q_init_316.mat' % args.matrix_dir

if os.path.exists(Qinit_Name):
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    X_data = Training_labels.transpose()
    # Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})

Y_data = Y_data.transpose()
Training_labels = np.concatenate((Training_labels, Y_data), axis=1)
# Valid_labels = Training_labels[nrtrain:,:]
Training_labels = Training_labels[:nrtrain, :]

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.cuda()

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.cuda()


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=8,
                         shuffle=True)

model = ISTA_Netpp(layer_num, n_output)
# model = nn.DataParallel(model)
model = model.cuda()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_ISTA_Net_pp_layer_%d_lr_%.4f_size_%d_time_%s" % (
    args.model_dir, layer_num, learning_rate, y_size, nowtime)

log_file_name = "./%s/Log_CS_ISTA_Net_pp_layer_%d_lr_%.4f_size_%d_time_%s.txt" % (
    args.log_dir, layer_num, learning_rate, y_size, nowtime)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

j = 0

print('trainset size: ', len(rand_loader))

print("-------------start_train------------\n")
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    for data in rand_loader:
        x = data[:, :16384].view(-1, 1, 128, 128).cuda()
        y = data[:, 16384:].cuda().unsqueeze(2).unsqueeze(3)
        # print('x.shape, y.shape, gamma.shape, Phi.shape, n_input\n', x.shape, y.shape, gamma.shape, Phi.shape, n_input)
        # print(type(x), type(y), type(gamma), type(Phi))
        optimizer.zero_grad()

        x_output = model(y, gamma, Phi, n_input)

        print('-------------output------------\n')
        st = time.time()
        Prediction_value = x_output.cpu().data.numpy()
        ed = time.time()
        print('pred time: ', ed - st)
        X_rec = np.reshape(Prediction_value, (-1, 128, 128))
        X_ori = np.reshape(x.cpu().data.numpy(), (-1, 128, 128))

        IMG = np.concatenate((np.clip(X_ori[0], 0, 1), np.clip(X_rec[0], 0, 1)), axis=1)
        writer.add_image('IMG', IMG, global_step=j, dataformats='HW')

        loss = torch.mean(torch.pow(x_output - x, 2))
        writer.add_scalar('loss_all', loss, global_step=j)
        j += 1

        loss.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.8f\n" % (epoch_i, end_epoch, loss.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
writer.close()
