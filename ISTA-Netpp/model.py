import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time


class ConditionModule(nn.Module):
    def __init__(self, layerNum):
        super(ConditionModule, self).__init__()
        model = [
            nn.Linear(1, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, layerNum * 2, bias=True),
            nn.Softplus(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        num = int(x.shape[1] / 2)
        return x[0, 0:num], x[0, num:]


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x0 = x[0]
        cond = x[1]
        out = self.act(self.conv1(x0))
        out = self.conv2(out)
        return x0 + out, cond


def PhiTPhi(x, PhiWeight, PhiTWeight):
    tmp = F.conv2d(x, PhiWeight, bias=None, stride=128, padding=0)
    tmp = F.conv2d(tmp, PhiTWeight, bias=None, padding=0)
    tmp = F.pixel_shuffle(tmp, upscale_factor=128)
    return tmp


class DGDMBlock(nn.Module):
    def __init__(self):
        super(DGDMBlock, self).__init__()

    def forward(self, x, y, rho, PhiWeight, PhiTWeight):
        # PhiWeight = Phi.contiguous().view(-1, 1, 33, 33)
        # PhiTWeight = Phi.transpose().contiguous().view(-1, 1, 1, 1)

        r = x - rho * PhiTPhi(x, PhiWeight, PhiTWeight)
        r = r + rho * y
        return r


class DPMMBlock(nn.Module):
    def __init__(self, nf=32):
        super(DPMMBlock, self).__init__()
        self.extr = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        self.rb1 = ResidualBlock(nf)
        self.rb2 = ResidualBlock(nf)
        self.recon = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

    def forward(self, r, sigma):
        m = sigma.repeat(r.shape[0], 1, r.shape[2], r.shape[3])
        in0 = torch.cat((r, m), 1)
        # print('M, in0\n', m.device, in0.device)
        out = self.extr(in0)
        out, _ = self.rb1([out, None])
        out, _ = self.rb2([out, None])
        out = self.recon(out)
        # print('r, out:', r.shape, out.shape)
        return r + out


class ISTA_Netpp(nn.Module):
    def __init__(self, layerNum, n_output):
        super(ISTA_Netpp, self).__init__()
        self.layerNum = layerNum
        self.n_output = n_output

        self.cm = ConditionModule(layerNum)
        blocks = []
        for i in range(layerNum):
            blocks.append(nn.ModuleList([DGDMBlock(), DPMMBlock()]))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, batch_y, gamma, Phi, n_input):
        # print('batch_y, gamma, Phi\n', batch_y.device, gamma.device, Phi.device)
        rho_k, sigma_k = self.cm(gamma)

        PhiWeight = Phi.contiguous().view(n_input, 1, 128, 128)
        PhiTWeight = Phi.t().contiguous().view(self.n_output, n_input, 1, 1)
        # y = F.conv2d(batch_x, PhiWeight, stride=33, padding=0, bias=None)
        y = F.conv2d(batch_y, PhiTWeight, padding=0, bias=None)
        y = nn.PixelShuffle(128)(y)
        x = y

        for i in range(self.layerNum):
            st = time.time()
            r = self.blocks[i][0](x, y, rho_k[i], PhiWeight, PhiTWeight)
            mid = time.time()
            # print(r.type, sigma_k.type)
            x = self.blocks[i][1](r, sigma_k[i])
            ed = time.time()
            print('time: ', ed - mid, mid - st)

        return x


if __name__ == '__main__':
    CM = ConditionModule(20)
    gamma = torch.from_numpy(np.arange(0.1, 0.5, 0.1, dtype=np.float32).reshape(1, -1))
    print(gamma[:, 0:1])
    print(CM(gamma[:, 0:1]))
