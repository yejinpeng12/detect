import torch
from YOLOv7.basebone.Conv import Conv
from torch import nn
class Downsample(nn.Module):
    def __init__(self,in_dim,out_dim,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.Pooling = nn.MaxPool2d((2,2),2)
        self.conv1 = Conv(in_dim,inter_dim,1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv2 = Conv(in_dim,inter_dim,1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.conv3 = Conv(inter_dim,inter_dim,3,s=2,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
    def forward(self,x):
        x1 = self.Pooling(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x)
        x4 = self.conv3(x3)

        return torch.cat([x2,x4],dim=1)
