import torch
import torch.nn as nn


# 下采样
def de_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
# 上采样
def up_conv_block(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class UNet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.de1 = de_conv_block(in_channels, 32)

        self.de2 = de_conv_block(32, 64)

        self.de3 = de_conv_block(64, 128)

        self.de4 = de_conv_block(128, 256)

        # 底部模块
        self.de5 = de_conv_block(256, 512)

        self.up4 = up_conv_block(512, 256)
        self.de6 = de_conv_block(512, 256)

        self.up3 = up_conv_block(256, 128)
        self.de7 = de_conv_block(256, 128)

        self.up2 = up_conv_block(128, 64)
        self.de8 = de_conv_block(128, 64)

        self.up1 = up_conv_block(64, 32)
        self.de9 = de_conv_block(64, 32)

        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        # 1, 32, 256, 256
        x1 = self.de1(x)
        # 1, 64, 128, 128
        x2 = self.de2(self.pool(x1))
        # 1, 128, 64, 64
        x3 = self.de3(self.pool(x2))
        # 1, 256, 32, 32
        x4 = self.de4(self.pool(x3))

        # 底部连接
        # 1, 512, 16, 16
        x5 = self.de5(self.pool(x4))

        # Decoding path
        # 1, 256, 32, 32
        x6 = self.up4(x5)
        # 1, 512, 32, 32
        x6 = torch.cat((x6, x4), dim=1)
        # 1, 256, 32, 32
        x6 = self.de6(x6)

        # 1, 128, 64, 64
        x7 = self.up3(x6)
        # 1, 256, 64, 64
        x7 = torch.cat((x7, x3), dim=1)
        # 1, 128, 64, 64
        x7 = self.de7(x7)

        # 1, 64, 128, 128
        x8 = self.up2(x7)
        # 1, 128, 128, 128
        x8 = torch.cat((x8, x2), dim=1)
        # 1, 64, 128, 128
        x8 = self.de8(x8)

        # 1, 32, 256, 256
        x9 = self.up1(x8)
        # 1, 64, 256, 256
        x9 = torch.cat((x9, x1), dim=1)
        # 1, 32, 256, 256
        x9 = self.de9(x9)

        return torch.sigmoid(self.output(x9))


model = UNet(in_channels=3, out_channels=1)
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 1 channel, 572x572 image size
output_tensor = model(input_tensor)
print(output_tensor.shape)
