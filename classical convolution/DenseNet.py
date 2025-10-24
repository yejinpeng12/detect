import torch.nn as nn
import torch
import numpy as np

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, 3, 1, 1, bias=False)
    )
#核心模块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels * i + input_channels, num_channels)
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, 1, 1, 0, bias=False),
        nn.AvgPool2d(2, 2)
    )

class DenseNet(nn.Module):
    def __init__(self, num_convs_in_dense_block, num_channels, growth_rate):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, num_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        layers = []
        for i, num_convs in enumerate(num_convs_in_dense_block):
            layers.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_block) - 1:
                layers.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.block2 = nn.Sequential(*layers)

        self.block3 = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 100)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

if __name__ == "__main__":
    c = np.ndarray([3, 3, 32, 32])
    c = torch.tensor(c, dtype=torch.float32).to('cuda')
    a = DenseNet([4, 4, 4, 4], 64, 32).to('cuda')
    print(a(c).shape)