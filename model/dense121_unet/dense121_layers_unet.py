# Architecture partially copied from:
# https://github.com/bamos/densenet.pytorch/blob/master/densenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, start=False):
        super(PreDenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.start = start

        self.start_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if start:
            self.feed_forward = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            )

    def forward(self, x):
        if self.start:
            x = s = self.start_conv(x)

            return self.feed_forward(x), s

        return self.feed_forward(x)


class DenseSingleLayer(nn.Module):
    def __init__(self, n_channels, growth_rate):
        super(DenseSingleLayer, self).__init__()
        self.bn = nn.BatchNorm2d(n_channels)
        self.conv = nn.Conv2d(n_channels, growth_rate, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat((x, out), 1)
        return out


class DenseTransition(nn.Module):
    def __init__(self, n_channels, n_out_channels, upsample):
        super(DenseTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1)

        self.upsample = upsample

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        if not self.upsample:
            out = F.avg_pool2d(out, 2)
        else:
            out = F.interpolate(out, scale_factor=2)

        return out


def make_dense(n_channels, growth_rate, n_dense_blocks):
    layers = []
    for i in range(int(n_dense_blocks)):
        layers.append(DenseSingleLayer(n_channels, growth_rate))
        n_channels += growth_rate
    return nn.Sequential(*layers)



class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_dense_blocks, k, transition=True, upsample=False):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.transition = transition

        self.dense = make_dense(n_channels=in_channels, growth_rate=k, n_dense_blocks=n_dense_blocks)

        if transition:
            self.trans = DenseTransition(in_channels + n_dense_blocks * k, out_channels, upsample)

    def forward(self, x):

        if self.transition:
            return self.trans(self.dense(x))

        return self.dense(x)



class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, double=False, intermezzo=False):
        super(ConvTBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if double:
            if not intermezzo:
                self.convt = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2, 2), stride=2)
                )
            else:
                self.convt = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=(2, 2), stride=2)
                )
        else:
            self.convt = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
            )

    def forward(self, x, s):
        x = self.convt(x)

        return torch.cat((x, s), dim=1)


if __name__ == '__main__':
    x = torch.randn((1, 64, 56, 56))

    nDenseBlocks = 6
    growthRate = 32
    nChannels = 64
    reduction = 0.5

    dense_block = DenseBlock(64, 256, 6, k=32)

    y = dense_block(x)

    print(y.shape)

