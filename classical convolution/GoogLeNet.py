import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block3 = nn.Sequential(
            Inception(192, 64, [96, 128], [16, 32], 32),
            Inception(256, 128, [128, 192], [32, 96], 64),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block4 = nn.Sequential(
            Inception(480, 192, [96, 208], [16, 48], 64),
            Inception(512, 160, [112, 224], [24, 64], 64),
            Inception(512, 128, [128, 256], [24, 64], 64),
            Inception(512, 112, [128, 288], [32, 64], 64),
            Inception(528, 256, [160, 320], [32, 128], 128),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block5 = nn.Sequential(
            Inception(832, 256, [160, 320], [32, 128], 128),
            Inception(832, 384, [192, 384], [48, 128], 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

        #参数初始化
        for m in self.modules():
            #卷积层参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #全连接参数初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class Inception(nn.Module):
    def __init__(self, input_channels, c1, c2, c3, c4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, c1, 1, 1, 0),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(input_channels, c2[0], 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3, 1, 1),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(input_channels, c3[0], 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5, 1, 2),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_channels, c4, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        y = torch.cat([x1, x2, x3, x4], 1)
        return y

if __name__ == "__main__":
    a = GoogLeNet(Inception).to('cuda')
    c = np.ndarray([16 , 3, 28, 28])
    c = torch.tensor(c, dtype=torch.float32).to("cuda")
    print(a(c).shape)