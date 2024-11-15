import torch
import torch.nn as nn

from layers import (
    unetConv,
    skipConPool,
    skipConSameHeight,
    skipConUpSamp
)


class UNET3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET3Plus, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.conv1 = unetConv(in_channels, filters[0])
        self.conv2 = unetConv(filters[0], filters[1])
        self.conv3 = unetConv(filters[1], filters[2])
        self.conv4 = unetConv(filters[2], filters[3])
        self.conv5 = unetConv(filters[3], filters[4])
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.CatChannels = filters[0]
        self.CatBlocks = len(filters)
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.x1e_to_x4d = skipConPool(filters[0], self.CatChannels, 8)
        self.x2e_to_x4d = skipConPool(filters[1], self.CatChannels, 4)
        self.x3e_to_x4d = skipConPool(filters[2], self.CatChannels, 2)
        self.x4e_to_x4d = skipConSameHeight(filters[3], self.CatChannels)
        self.x5e_to_x4d = skipConUpSamp(filters[4], self.CatChannels, 2)
        self.conv_x4d = unetConv(self.UpChannels, self.UpChannels)

        self.x1e_to_x3d = skipConPool(filters[0], self.CatChannels, 4)
        self.x2e_to_x3d = skipConPool(filters[1], self.CatChannels, 2)
        self.x3e_to_x3d = skipConSameHeight(filters[2], self.CatChannels)
        self.x4d_to_x3d = skipConUpSamp(self.UpChannels, self.CatChannels, 2)
        self.x5d_to_x3d = skipConUpSamp(filters[4], self.CatChannels, 4)
        self.conv_x3d = unetConv(self.UpChannels, self.UpChannels)

        self.x1e_to_x2d = skipConPool(filters[0], self.CatChannels, 2)
        self.x2e_to_x2d = skipConSameHeight(filters[1], self.CatChannels)
        self.x3d_to_x2d = skipConUpSamp(self.UpChannels, self.CatChannels, 2)
        self.x4d_to_x2d = skipConUpSamp(self.UpChannels, self.CatChannels, 4)
        self.x5d_to_x2d = skipConUpSamp(filters[4], self.CatChannels, 8)
        self.conv_x2d = unetConv(self.UpChannels, self.UpChannels)

        self.x1e_to_x1d = skipConSameHeight(filters[0], self.CatChannels)
        self.x2d_to_x1d = skipConUpSamp(self.UpChannels, self.CatChannels, 2)
        self.x3d_to_x1d = skipConUpSamp(self.UpChannels, self.CatChannels, 4)
        self.x4d_to_x1d = skipConUpSamp(self.UpChannels, self.CatChannels, 8)
        self.x5d_to_x1d = skipConUpSamp(filters[4], self.CatChannels, 16)
        self.conv_x1d = unetConv(self.UpChannels, self.UpChannels)

        # output
        self.final_conv = nn.Conv2d(self.UpChannels, out_channels, 3, padding=1)

    def forward(self, x):
        h1 = self.conv1(x)

        h2 = self.pool(h1)
        h2 = self.conv2(h2)

        h3 = self.pool(h2)
        h3 = self.conv3(h3)

        h4 = self.pool(h3)
        h4 = self.conv4(h4)

        h5 = self.pool(h4)
        h5 = self.conv5(h5)

        x4d_cat = torch.cat((
            self.x1e_to_x4d(h1),
            self.x2e_to_x4d(h2),
            self.x3e_to_x4d(h3),
            self.x4e_to_x4d(h4),
            self.x5e_to_x4d(h5)
        ), 1)
        x4d = self.conv_x4d(x4d_cat)

        x3d_cat = torch.cat((
            self.x1e_to_x3d(h1),
            self.x2e_to_x3d(h2),
            self.x3e_to_x3d(h3),
            self.x4d_to_x3d(x4d),
            self.x5d_to_x3d(h5)
        ), 1)
        x3d = self.conv_x3d(x3d_cat)

        x2d_cat = torch.cat((
            self.x1e_to_x2d(h1),
            self.x2e_to_x2d(h2),
            self.x3d_to_x2d(x3d),
            self.x4d_to_x2d(x4d),
            self.x5d_to_x2d(h5)
        ), 1)
        x2d = self.conv_x2d(x2d_cat)

        x1d_cat = torch.cat((
            self.x1e_to_x1d(h1),
            self.x2d_to_x1d(x2d),
            self.x3d_to_x1d(x3d),
            self.x4d_to_x1d(x4d),
            self.x5d_to_x1d(h5)
        ), 1)
        x1d = self.conv_x1d(x1d_cat)

        x = self.final_conv(x1d)

        return torch.sigmoid(x)



if __name__ == "__main__":
    unet3Plus = UNET3Plus().to("cuda")
    x = torch.Tensor(1, 3, 1024, 1024).to("cuda")

    out = unet3Plus(x)

    print(out.shape)
