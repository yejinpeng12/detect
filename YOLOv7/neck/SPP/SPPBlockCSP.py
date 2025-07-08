from torch import nn
from YOLOv7.basebone.Conv import Conv
from SPPF import SPPF
import torch
class SPPFBlockCSP(nn.Module):
    def __init__(self,in_dim,out_dim,pooling_size=5,expand_radio=0.5,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        inter_dim = int(in_dim * expand_radio)
        self.conv1 = Conv(in_dim,inter_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.conv2 = Conv(in_dim,inter_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.conv3 = nn.Sequential(
            Conv(inter_dim,inter_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            SPPF(inter_dim,inter_dim,expand_radio=1.0,pooling_size=pooling_size,norm_type=norm_type,act_type=act_type),
            Conv(inter_dim,inter_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )
        self.conv4 = Conv(inter_dim * 2,out_dim,k=1,norm_type=norm_type,act_type=act_type)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        return self.conv4(torch.cat([x1,x3],dim=1))
