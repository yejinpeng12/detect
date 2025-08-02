from YOLOv7.basebone.Conv import Conv
from YOLOv7.neck.PaFPN.ELANBlockPaFPN import ELANBlockFPN
from torch import nn
from YOLOv7.basebone.Downsample import Downsample
import torch.nn.functional as F
import torch
class YOLOv7PaFPN(nn.Module):
    def __init__(self,in_dims,out_dim,channel_width = 0.5,norm_type='BN',act_type='silu',depthwise=False):
        super().__init__()
        self.in_dims = in_dims
        self.channel_width = channel_width
        c3,c4,c5 = in_dims

        self.reduce_layer_1 = Conv(c5, round(256*channel_width), k=1, norm_type=norm_type, act_type=act_type)
        self.reduce_layer_2 = Conv(c4 , round(256*channel_width), k=1, norm_type=norm_type, act_type=act_type)
        self.top_down_layer_1 = ELANBlockFPN(in_dim=round(256*channel_width) + round(256*channel_width),out_dim=round(256*channel_width),act_type=act_type,norm_type=norm_type,depthwise=depthwise)

        self.reduce_layer_3 = Conv(round(256*channel_width),  round(128*channel_width), k=1, norm_type=norm_type,act_type=act_type)
        self.reduce_layer_4 = Conv(c3,  round(128*channel_width), k=1, norm_type=norm_type,act_type=act_type)
        self.top_down_layer_2 = ELANBlockFPN(in_dim=round(128*channel_width) + round(128*channel_width),out_dim=round(128*channel_width),norm_type=norm_type,act_type=act_type,depthwise=depthwise)

        self.downsample_layer_1 = Downsample(in_dim=round(128*channel_width),out_dim=round(256*channel_width),norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.bottom_up_layer_1 = ELANBlockFPN(in_dim=round(256*channel_width) + round(256*channel_width),out_dim=round(256*channel_width),norm_type=norm_type,act_type=act_type,depthwise=depthwise)

        self.downsample_layer_2 = Downsample(in_dim=round(256*channel_width),out_dim= round(512*channel_width),norm_type=norm_type,act_type=act_type,depthwise=depthwise)
        self.bottom_up_layer_2 = ELANBlockFPN(in_dim=round(512*channel_width)+c5,out_dim=round(512*channel_width),norm_type=norm_type,act_type=act_type,depthwise=depthwise)

        self.head_conv_1 = Conv(round(128*channel_width), round(256*channel_width),k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=False)
        self.head_conv_2 = Conv(round(256*channel_width), round(512*channel_width),k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=False)
        self.head_conv_3 = Conv(round(512*channel_width), round(1024*channel_width),k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=False)

        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1, norm_type=norm_type,act_type=act_type) for in_dim in [round(256*channel_width), round(512*channel_width), round(1024*channel_width)]
            ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [round(256*channel_width), round(512*channel_width), round(1024*channel_width)]
    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6,scale_factor=2.0)#上采样
        c8 = torch.cat([self.reduce_layer_2(c4),c7],dim=1)
        c9 = self.top_down_layer_1(c8)

        c10 = self.reduce_layer_3(c9)
        c11 = F.interpolate(c10,scale_factor=2.0)
        c12 = torch.cat([self.reduce_layer_4(c3),c11],dim=1)
        c13 = self.top_down_layer_2(c12)

        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c9,c14],dim=1)
        c16 = self.bottom_up_layer_1(c15)

        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c5,c17],dim=1)
        c19 = self.bottom_up_layer_2(c18)

        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)

        out_feats = [c20,c21,c22]

        if self.out_layers is not None:
            out_feats_proj = []
            for feat,layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj
        return out_feats

# c3, c4, c5 = torch.randn(10,512,80,80),torch.randn(10,1024,40,40),torch.randn(10,1024,20,20)
# feature = [c3,c4,c5]
# in_dims = [c3.shape[1],c4.shape[1],c5.shape[1]]
# module = YOLOv7PaFPN(in_dims=in_dims,out_dim=100)
# for i in module(feature):
#     print(i.shape)

