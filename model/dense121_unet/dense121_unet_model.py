import torch
import torch.nn as nn


from dense121_layers_unet import PreDenseBlock, DenseBlock, ConvTBlock


class DenseNet121(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, k=32):
        super(DenseNet121, self).__init__()

        self.pd1 = PreDenseBlock(in_channels, 64, (7, 7), 2, 3, start=True)
        self.pd2 = PreDenseBlock(256, 128, (1, 1), 1, 0)
        self.pd3 = PreDenseBlock(512, 256, (1, 1), 1, 0)
        self.pd4 = PreDenseBlock(1024, 512, (1, 1), 1, 0)

        self.dn1 = DenseBlock(64, 256, 6, k)
        self.dn2 = DenseBlock(128, 512, 12, k)
        self.dn3 = DenseBlock(256, 1024, 24, k)
        self.dn4 = DenseBlock(512, 1024, 16, k, transition=False)

        self.dc1 = ConvTBlock(1024, 1024)
        self.dc2 = ConvTBlock(1024, 512, double=True)
        self.dc3 = ConvTBlock(512, 256, double=True)
        self.dc4 = ConvTBlock(256, 64, double=True, intermezzo=True)

        self.dn5 = DenseBlock(256, 1024, 24, k, upsample=True)
        self.dn6 = DenseBlock(128, 512, 12, k, upsample=True)
        self.dn7 = DenseBlock(64, 256, 6, k, upsample=True)

        self.decode_conv1 = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.decode_conv2 = nn.Conv2d(1024, 128, kernel_size=(1, 1))
        self.decode_conv3 = nn.Conv2d(512, 64, kernel_size=(1, 1))
        self.decode_conv4 = nn.Conv2d(128, 64, kernel_size=(7, 7), padding=3)

        self.final_block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(1, 1))
        )


    def forward(self, x):
        # Encoder
        x, s1 = self.pd1(x)

        x = s2 = self.dn1(x)

        x = self.pd2(x)
        x = s3 = self.dn2(x)

        x = self.pd3(x)
        x = s4 = self.dn3(x)

        x = self.pd4(x)
        x = self.dn4(x)


        # Decoder
        x = self.dc1(x, s4)
        x = self.decode_conv1(x)
        x = self.dn5(x)

        x = self.dc2(x, s3)
        x = self.decode_conv2(x)
        x = self.dn6(x)

        x = self.dc3(x, s2)
        x = self.decode_conv3(x)
        x = self.dn7(x)

        x = self.dc4(x, s1)
        x = self.decode_conv4(x)

        x = self.final_block(x)

        return x



if __name__ == '__main__':
    x = torch.randn((1, 3, 1024, 1024)).cuda()

    model = DenseNet121().cuda()

    y = model(x)

    print(y.shape)
