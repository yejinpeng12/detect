import torch.nn as nn
from basic_block import Concat, A2C2f, Conv, C3k2

class neck(nn.Module):
    def __init__(self, in_dim, depth=0.5, width=0.5):#0.5
        super().__init__()
        self.out_dim = [round(256 * width), round(512 * width), round(1024 * width)+in_dim[2]]
        self.up_1 = nn.Upsample(None, 2, "nearest")
        self.concat_1 = Concat(1)
        self.a2c2f_1 = A2C2f(c1=in_dim[1] + in_dim[2], c2=round(512 * width), n=round(2 * depth), a2=False, area=-1)

        self.up_2 = nn.Upsample(None, 2, "nearest")
        self.concat_2 = Concat(1)
        self.a2c2f_2 = A2C2f(c1=round(512 * width)+in_dim[0], c2=round(256 * width), n=round(2 * depth), a2=False, area=-1)

        self.conv1 = Conv(c1=round(256 * width), c2=round(256 * width), k=3, s=2)
        self.concat_3 = Concat(1)
        self.a2c2f_3 = A2C2f(c1=round(512 * width)+round(256 * width), c2=round(512 * width), n=round(2 * depth), a2=False, area=-1)

        self.conv2 = Conv(c1=round(512 * width),c2=round(512 * width), k=3, s=2)
        self.concat_4 = Concat(1)
        self.c3k2 = C3k2(c1=round(512 * width) + in_dim[2], c2=round(1024 * width)+in_dim[2], shortcut=True)
    def forward(self, x):
        c4, c6, c8 = x
        c9 = self.up_1(c8)
        c10 = self.concat_1([c9, c6])
        c11 = self.a2c2f_1(c10)

        c12 = self.up_2(c11)
        c13 = self.concat_2([c12, c4])
        c14 = self.a2c2f_2(c13)

        c15 = self.conv1(c14)
        c16 = self.concat_3([c15, c11])
        c17 = self.a2c2f_3(c16)

        c18 = self.conv2(c17)
        c19 = self.concat_4([c18, c8])
        c20 = self.c3k2(c19)

        return [c14, c17, c20]