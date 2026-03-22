from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.ReLU=nn.ReLU()
        self.c1 = nn.Conv2d(1, 96, kernel_size=11, stride=4)
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.p4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.p8 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten=nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)



    def forward(self, x):
        # conv + relu + pool
        x = self.ReLU(self.c1(x))
        x = self.p2(x)
        x = self.ReLU(self.c3(x))
        x = self.p4(x)

        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.p8(x)

        x = self.flatten(x)

        x = self.ReLU(self.fc1(x))
        x = F.dropout(x,0.5)
        x = self.ReLU(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        return x




# 测试
if __name__ == "__main__":
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model,(1,227,227)))

