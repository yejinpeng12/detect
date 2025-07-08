from torch import nn
import torch
from YOLOv7.basebone.Conv import Conv
class ELANBlockFPN(nn.Module):
    def __init__(self,in_dim,out_dim,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        self.conv1 = Conv(in_dim,in_dim//2,k=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv2 = Conv(in_dim,in_dim//2,k=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv3 = Conv(in_dim//2,in_dim//4,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv4 = Conv(in_dim//4,in_dim//4,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv5 = Conv(in_dim // 4, in_dim // 4, k=3, p=1, norm_type=norm_type, act_type=act_type,
                          depthwise=depthwise)
        self.conv6 = Conv(in_dim // 4, in_dim // 4, k=3, p=1, norm_type=norm_type, act_type=act_type,
                          depthwise=depthwise)
        self.conv7 = Conv(in_dim * 2,out_dim,k=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return self.conv7(torch.cat([x1,x2,x3,x4,x5,x6],dim=1))
