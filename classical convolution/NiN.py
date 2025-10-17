import torch.nn as nn
import torch.nn.functional as F

def nin_block(input_channels, output_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, pad),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, 1, 1, 0),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, 1, 1, 0)
    )

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nin_block(3, 96, 11, 4, 0)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.block2 = nin_block(96, 256, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.block3 = nin_block(256, 384, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(3, 2)
        self.drop = nn.Dropout(0.5)
        self.block4 = nin_block(384, 100, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        #参数初始化
        for m in self.modules():
            #卷积层参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.drop(self.pool3(self.block3(x)))
        x = self.avg_pool(self.block4(x))
        x = self.flatten(x)
        return x