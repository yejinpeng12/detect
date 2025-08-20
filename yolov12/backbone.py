from basic_block import Conv, C3k2, A2C2f
import torch.nn as nn

class BackBone(nn.Module):
    def __init__(self, depth=0.5, width=0.25):#0.5
        super().__init__()
        self.out_dim = [round(512 * width), round(512 * width), round(1024 * width)]
        self.conv1 = Conv(c1=4, c2=round(64 * width), k=3, s=2)
        self.conv2 = Conv(c1=round(64 * width), c2=round(128 * width), k=3, s=2)
        self.c3k2_1 = C3k2(c1=round(128 * width), c2=round(256 * width), n=round(2 * depth), shortcut=False, e=0.25)
        self.conv3 = Conv(c1=round(256 * width), c2=round(256 * width), k=3, s=2)
        self.c3k2_2 = C3k2(c1=round(256 * width), c2=round(512 * width), n=round(2 * depth), shortcut=False, e=0.25)
        self.conv4 = Conv(c1=round(512 * width), c2=round(512 * width), k=3, s=2)
        self.a2c2f_1 = A2C2f(c1=round(512 * width), c2=round(512 * width), n=round(4 * depth), shortcut=True, area=4)
        self.conv5 = Conv(c1=round(512 * width), c2=round(1024 * width), k=3, s=2)
        self.a2c2f_2 = A2C2f(c1=round(1024 * width), c2=round(1024 * width), n=round(4 * depth), shortcut=True, area=1)
    def forward(self, x):
        c0 = self.conv1(x)
        c1 = self.conv2(c0)
        c2 = self.c3k2_1(c1)
        c3 = self.conv3(c2)
        c4 = self.c3k2_2(c3)
        c5 = self.conv4(c4)
        c6 = self.a2c2f_1(c5)
        c7 = self.conv5(c6)
        c8 = self.a2c2f_2(c7)

        return [c4, c6, c8]