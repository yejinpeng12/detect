from torch import nn
from YOLOv7.basebone.Conv import Conv
from YOLOv7.basebone.ELANBlock import ELANBlock
from YOLOv7.basic_block_v7.simAM_module import simam_module

class ELANNet_Tiny(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_Tiny, self).__init__()
        # -------------- Basic parameters --------------
        self.feat_dims = [32, 64, 128, 256, 512]
        #self.squeeze_ratios = [0.5, 0.5, 0.5, 0.5]  # Stage-1 -> Stage-4
        self.squeeze_ratios = [1.0,1.0,1.0,0.5]#[1.0,1.0,1.0,1.0]
        self.branch_depths = [2, 2, 2, 2]  # Stage-1 -> Stage-4

        # -------------- Network parameters --------------
        ## P1/2
        self.layer_1 = Conv(4, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type,
                            depthwise=depthwise)
        ## P2/4: Stage-1
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type,
                 depthwise=depthwise),
            ELANBlock(self.feat_dims[1], self.feat_dims[1], self.squeeze_ratios[0], self.branch_depths[0],
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P3/8: Stage-2
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            ELANBlock(self.feat_dims[1], self.feat_dims[2], self.squeeze_ratios[1], self.branch_depths[1],
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P4/16: Stage-3
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            ELANBlock(self.feat_dims[2], self.feat_dims[3], self.squeeze_ratios[2], self.branch_depths[2],
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        ## P5/32: Stage-4
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            ELANBlock(self.feat_dims[3], self.feat_dims[4], self.squeeze_ratios[3], self.branch_depths[3],
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs