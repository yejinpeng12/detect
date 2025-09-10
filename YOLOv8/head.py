from torch import nn
from Conv import Conv
from dfl import DFL

class head(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes=5, num_cls_head=2, num_reg_head=2, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        #类别检测头
        cls_feats = []
        for i in range(num_cls_head):
            cls_feats.append(Conv(in_dim, out_dim, k=3, p=1))
        cls_feats.append(nn.Conv2d(out_dim, num_classes, 1))
        #回归检测头
        reg_feats = []
        for i in range(num_reg_head):
            reg_feats.append(Conv(in_dim, out_dim, k=3, p=1))
        reg_feats.append(nn.Conv2d(out_dim, 4 * reg_max, 1))
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)
    def forward(self,x):
        cls_outputs = self.cls_feats(x)#(B, self.num_classes, H, W)
        reg_outputs = self.reg_feats(x)#(B, 4*self.reg_max, H, W)
        return cls_outputs, reg_outputs