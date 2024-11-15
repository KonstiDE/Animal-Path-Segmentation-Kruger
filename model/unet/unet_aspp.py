import torch
from torch import nn
from torch.nn import functional as F


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPPConv, self).__init__()

        self.aspp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d()
        )

    def forward(self, x):
        return self.aspp_conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()

        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)



class ASPP(nn.Module):
    def __init__(self, atrous_rates, in_channels=512, out_channels=512):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]

        for rate in tuple(atrous_rates):
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # One big convolution to
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)

        return self.project(res)
