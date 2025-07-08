from torch import nn
from YOLOv7.basebone.Conv import Conv
from YOLOv7.basebone.ELANBlock import ELANBlock
from YOLOv7.basebone.Downsample import Downsample
class ELANNet(nn.Module):
    #k=3,p=1,s=2会将图片长宽减半
    def __init__(self,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        self.feat_dim = [32,64,128,256,512,1024,1024]
        self.squeeze_ratios = [0.5,0.5,0.5,0.25]
        self.branch_depths = [2,2,2,2]

        self.layer1 = nn.Sequential(
            Conv(4, 32, k=3, p=1, norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            Conv(32, 64, k=3, p=1, s=2, norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            Conv(64,64, k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )

        self.layer2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, norm_type=norm_type, act_type=act_type, depthwise=depthwise),
            ELANBlock(in_dim=128,out_dim=256,expand_radio=self.squeeze_ratios[0],branch_depths=self.branch_depths[0],norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )

        self.layer3 = nn.Sequential(
            Downsample(in_dim=256,out_dim=256,norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            ELANBlock(in_dim=256,out_dim=512,expand_radio=self.squeeze_ratios[1],branch_depths=self.branch_depths[1],norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )

        self.layer4 = nn.Sequential(
            Downsample(in_dim=512,out_dim=512,norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            ELANBlock(in_dim=512,out_dim=1024,expand_radio=self.squeeze_ratios[2],branch_depths=self.branch_depths[2],norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )

        self.layer5 = nn.Sequential(
            Downsample(in_dim=1024,out_dim=1024,norm_type=norm_type,act_type=act_type,depthwise=depthwise),
            ELANBlock(in_dim=1024,out_dim=1024,expand_radio=self.squeeze_ratios[3],branch_depths=self.branch_depths[3],norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        )
    def forward(self,x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return [c3,c4,c5]

