from base_block import SPPF,C2f
from Conv import Conv
import torch
from torch import nn

class PaFPN(nn.Module):
    def __init__(self, in_dim, depth=0.66, width=0.25):
        super().__init__()
        self.out_dim = [round(256 * width), round(512 * width), round(1024 * width)]
        self.up_1 = nn.Upsample(None,2,"nearest")
        self.c2f_1 = C2f(in_c=in_dim[2] + in_dim[1], out_c=round(512 * width), n=round(3 * depth))

        self.up_2 = nn.Upsample(None,2,"nearest")
        self.c2f_2 = C2f(in_c=round(512 * width) + in_dim[0], out_c=round(256 * width), n=round(3 * depth))

        self.conv1 = Conv(in_c=round(256 * width), out_c=round(256 * width), k=3, s=2)
        self.c2f_3 = C2f(in_c=round(256 * width) + round(512 * width), out_c=round(512 * width), n=round(3 * depth))

        self.conv2 = Conv(in_c=round(512 * width), out_c=round(512 * width), k=3, s=2)
        self.c2f_4 = C2f(in_c=round(512 * width) + in_dim[2],out_c=round(1024 * width), n=round(3 * depth))
    def forward(self, x):
        x10 = self.up_1(x[2])
        x11 = torch.cat([x10, x[1]], dim=1)
        x12 = self.c2f_1(x11)

        x13 = self.up_2(x12)
        x14 = torch.cat([x13, x[0]], dim=1)
        x15 = self.c2f_2(x14)

        x16 = self.conv1(x15)
        x17 = torch.cat([x16, x12], dim=1)
        x18 = self.c2f_3(x17)

        x19 = self.conv2(x18)
        x20 = torch.cat([x19, x[2]], dim=1)
        x21 = self.c2f_4(x20)
        return [x15, x18, x21]#(B,C,H,W)