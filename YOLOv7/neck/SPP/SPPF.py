from torch import nn
from YOLOv7.basebone.Conv import Conv
import torch
class SPPF(nn.Module):
    def __init__(self,in_dim,out_dim,pooling_size=5,expand_radio=0.5,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        inter_dim = int(in_dim * expand_radio)
        self.out_dim = out_dim
        self.conv1 = Conv(in_dim,inter_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.conv2 = Conv(inter_dim*4,out_dim,k=1,norm_type=norm_type,act_type=act_type)
        self.max_pooling = nn.MaxPool2d(pooling_size,stride=1,padding=pooling_size//2)
    def forward(self,x):
        x = self.conv1(x)
        x1 = self.max_pooling(x)
        x2 = self.max_pooling(x1)
        x3 = self.max_pooling(x2)
        return self.conv2(torch.cat([x,x1,x2,x3],dim=1))
