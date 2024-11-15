import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetConv, self).__init__()
        self.unetConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unetConv(x)


class skipConPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scale):
        super(skipConPool, self).__init__()
        self.skipConPool = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_scale, stride=pool_scale, ceil_mode=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.skipConPool(x)


class skipConSameHeight(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(skipConSameHeight, self).__init__()
        self.skipConSameHeight = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.skipConSameHeight(x)


class skipConUpSamp(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale):
        super(skipConUpSamp, self).__init__()
        self.skipConUpSamp = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=up_scale),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.skipConUpSamp(x)
