# -*- coding: utf-8 -*-
import torch  # pytorch 0.4.0! fft
import torch.nn as nn
import numpy as np

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class CACFNet(nn.Module):

    def __init__(self, config=None):
        super(CACFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0
        self.config = config
        self.flag = False  # True for train, Flase for interference

    def forward(self, x):
        if self.flag:
            # self.feature.feature[0].padding = (0, 0)
            # self.feature.feature[2].padding = (0, 0)
            x = self.feature(x)
        else:
            # self.feature.feature[0].padding = (1, 1)
            # self.feature.feature[2].padding = (1, 1)
            x = self.feature(x) * self.config.cos_window
        xf = torch.rfft(x, signal_ndim=2)
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.irfft(complex_mul(kxzf, self.model_alphaf), signal_ndim=2)
        return response


    def CACF_update(self, inputs, lr=1.):

        inputs[0] = self.feature(inputs[0]) * self.config.cos_window
        inputs[1] = self.feature(inputs[1]) * self.config.cos_window
        inputs[2] = self.feature(inputs[2]) * self.config.cos_window
        inputs[3] = self.feature(inputs[3]) * self.config.cos_window
        inputs[4] = self.feature(inputs[4]) * self.config.cos_window
        zf = torch.rfft(inputs[0], signal_ndim=2)  # target region
        cf1 = torch.rfft(inputs[1], signal_ndim=2)  # contex 1 region
        cf2 = torch.rfft(inputs[2], signal_ndim=2)  # contex 2 region
        cf3 = torch.rfft(inputs[3], signal_ndim=2)  # contex 3 region
        cf4 = torch.rfft(inputs[4], signal_ndim=2)  # contex 4 region
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kccf1 = torch.sum(torch.sum(cf1 ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kccf2 = torch.sum(torch.sum(cf2 ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kccf3 = torch.sum(torch.sum(cf3 ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kccf4 = torch.sum(torch.sum(cf4 ** 2, dim=4, keepdim=True), dim=1, keepdim=True)

        if lr > 0.99:
            alphaf = self.config.yf / (kzzf + self.config.lambda0)
        else:
            alphaf = self.config.yf / (kzzf + self.config.lambda0 + self.config.lambda1 * (kccf1 + kccf2 + kccf3 + kccf4))

        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data




    def load_param(self, net_path):
        checkpoint = torch.load(net_path)
        # for param
        self.feature.feature[0].weight.data = checkpoint['feature.0.weight'].data
        self.feature.feature[0].bias.data = checkpoint['feature.0.bias'].data
        self.feature.feature[2].weight.data = checkpoint['feature.2.weight'].data
        self.feature.feature[2].bias.data = checkpoint['feature.2.bias'].data
        # for checkpoint
        # self.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':

    # network test
    pass



