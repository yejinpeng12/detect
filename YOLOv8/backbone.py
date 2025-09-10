from base_block import SPPF,C2f
from Conv import Conv
from torch import nn

class Base_bone(nn.Module):
    def __init__(self, depth=0.66, width=0.5):
        super().__init__()
        self.feat_dim = [round(256 * width), round(512 * width), round(1024 * width)]
        self.conv1 = Conv(in_c=4, out_c=round(64 * width), k=3, s=2)
        self.conv2 = Conv(in_c=round(64 * width), out_c=round(128 * width), k=3, s=2)
        self.c2f_1 = C2f(in_c=round(128 * width), out_c=round(128 * width), n=round(3 * depth),shortcut=True)
        self.conv3 = Conv(in_c=round(128 * width), out_c=round(256 * width), k=3, s=2)
        self.c2f_2 = C2f(in_c=round(256 * width), out_c=round(256 * width), n=round(6 * depth),shortcut=True)
        self.conv4 = Conv(in_c=round(256 * width), out_c=round(512 * width), k=3, s=2)
        self.c2f_3 = C2f(in_c=round(512 * width), out_c=round(512 * width), n=round(6 * depth),shortcut=True)
        self.conv5 = Conv(in_c=round(512 * width), out_c=round(1024 * width), k=3, s=2)
        self.c2f_4 = C2f(in_c=round(1024 * width), out_c=round(1024 * width), n=round(3 * depth),shortcut=True)
        self.SPPF = SPPF(in_c=round(1024 * width), out_c=round(1024 * width), k=5)
    def forward(self,x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.c2f_1(x1)
        x3 = self.conv3(x2)
        x4 = self.c2f_2(x3)
        x5 = self.conv4(x4)
        x6 = self.c2f_3(x5)
        x7 = self.conv5(x6)
        x8 = self.c2f_4(x7)
        x9 = self.SPPF(x8)
        return [x4, x6, x9]

