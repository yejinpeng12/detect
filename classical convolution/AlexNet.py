import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 96, 11, 4, 0)
        self.max_pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.max_pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)

        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)

        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.max_pool3 = nn.MaxPool2d(3, 2)

        self.linear1 = nn.Linear(6*6*256, 4096)

        self.linear2 = nn.Linear(4096, 4096)

        self.linear3 = nn.Linear(4096, 100)

        #参数初始化
        for m in self.modules():
            #卷积层参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #全连接参数初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.max_pool1(F.relu(self.conv1(x)))
        x = self.max_pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 6*6*256)
        x = F.dropout(F.relu(self.linear1(x)), 0.5)
        x = F.dropout(F.relu(self.linear2(x)), 0.5)
        x = self.linear3(x)

        return x
if __name__ == "__main__":
    c = np.ndarray([16, 3, 227, 227])
    c = torch.tensor(c, dtype=torch.float32).to('cuda')
    a = AlexNet().to('cuda')
    print(a(c).shape)