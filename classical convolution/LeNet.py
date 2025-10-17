import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.avg_pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.avg_pool2 = nn.AvgPool2d(2, 2)

        self.linear1 = nn.Linear(16 * 5 * 5, 120)

        self.linear2 = nn.Linear(120, 84)

        self.linear3 = nn.Linear(84, 10)

        #参数初始化
        for m in self.modules():
            #卷积层参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #全连接参数初始化
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avg_pool1(F.sigmoid(self.conv1(x)))
        x = self.avg_pool2(F.sigmoid(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        x = self.linear3(x)

        return x
if __name__ == "__main__":
    c = np.ndarray([16, 1, 28, 28])
    c = torch.tensor(c, dtype=torch.float32).to('cuda')
    a = LeNet().to('cuda')
    print(a(c).shape)