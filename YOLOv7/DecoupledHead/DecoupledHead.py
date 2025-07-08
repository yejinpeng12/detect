from torch import nn
from YOLOv7.basebone.Conv import Conv
class DecoupledHead(nn.Module):
    def __init__(self,in_dim,out_dim,num_cls_head=2,num_reg_head=2,norm_type='BN',act_type='silu',num_classes=20,depthwise=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_reg_head=num_reg_head
        self.num_cls_head=num_cls_head
        self.norm_type = norm_type
        self.act_type = act_type
        #类别检测头
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(self.num_cls_head):
            if i==0:
                cls_feats.append(Conv(in_dim,self.cls_out_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise))
            else:
                cls_feats.append(Conv(self.cls_out_dim,self.cls_out_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise))
        #回归检测头
        reg_feats = []
        self.reg_out_dim = max(out_dim,64)
        for i in range(self.num_reg_head):
            if i==0:
                reg_feats.append(Conv(in_dim,self.reg_out_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise))
            else:
                reg_feats.append(Conv(self.reg_out_dim,self.reg_out_dim,k=3,p=1,norm_type=norm_type,act_type=act_type,depthwise=depthwise))
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)
    def forward(self,x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
        return cls_feats, reg_feats
# # 输入特征的参数
# batch_size   = 2
# feat_channel = 512
# feat_height  = 20
# feat_width   = 20
#
# # 随机生成一张图像
# feature = torch.randn(batch_size, feat_channel, feat_height, feat_width)
# module = DecoupledHead(512,30,norm_type='BN',act_type='silu',num_cls_head=2,num_reg_head=2)
# for i in module(feature):
#     print(i.shape)