from YOLOv7.basebone.Conv import Conv
from torch import nn
import torch
#k=3,s=1,p=1不会改变长和宽。k=1,s=1,p=0不会改变长和宽
class ELANBlock(nn.Module):
    def __init__(self,in_dim,out_dim,expand_radio=0.5,branch_depths=2,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        inter_dim = int(in_dim * expand_radio)
        self.conv1 = Conv(in_dim,inter_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.conv2 = Conv(in_dim,inter_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.conv3 = nn.Sequential(*[
            Conv(inter_dim,inter_dim,k=3,s=1,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
            for _ in range(round(branch_depths))
        ])
        self.conv4 = nn.Sequential(*[
            Conv(inter_dim,inter_dim,k=3,s=1,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
            for _ in range(round(branch_depths))
                                     ])
        self.out = Conv(inter_dim * 4,out_dim,k=1,norm_type=norm_type,act_type=act_type)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x1)
        x4 = self.conv4(x3)
        return self.out(torch.cat([x1,x2,x3,x4],dim=1))

