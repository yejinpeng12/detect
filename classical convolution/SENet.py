import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        #压缩:全局平均池化,将每个通道的H*W特征图压缩为1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #激励:两个全连结层，学习通道间的相关性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),# 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),# 恢复维度
            nn.Sigmoid()# 将权重限制在0~1之间
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        #squeeze
        y = self.avg_pool(x).view(b, c)
        #Excitation
        y = self.fc(y).view(b, c, 1, 1)
        #scale:将学习到的权重乘以原始特征图
        return x * y.expand_as(x)

class SENet(nn.Module):
    def __init__(self, Residual):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block2 = nn.Sequential(
            Residual(64, 64, False, 1),
            Residual(64, 64, False, 1),
        )

        self.block3 = nn.Sequential(
            Residual(64, 128, True, 2),
            Residual(128, 128, False, 1)
        )

        self.block4 = nn.Sequential(
            Residual(128, 256, True, 2),
            Residual(256, 256, False, 1)
        )

        self.block5 = nn.Sequential(
            Residual(256, 512, True, 2),
            Residual(512, 512, False, 1)
        )

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

class SEResidual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, 1, stride, 0)
        else:
            self.conv3 = None

        self.se_block = SELayer(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        y = self.se_block(y)
        y = F.relu(y + x)
        # y = self.se_block(y) 放在后面也行
        return y

if __name__ == "__main__":
    c = np.ndarray([16, 3, 224, 224])
    c = torch.tensor(c, dtype=torch.float32).to('cuda')
    a = SENet(SEResidual).to('cuda')
    print(a(c).shape)