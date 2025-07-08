from torch import nn
def get_norm(norm_type,dim):
    if norm_type=='BN':
        return nn.BatchNorm2d(dim)
    elif norm_type=='GN':
        return nn.GroupNorm(32,dim)
    elif norm_type is not None:
        return nn.Identity()
def get_act(act_type):
    if act_type=='relu':
        return nn.ReLU(inplace=True)
    elif act_type=='lrelu':
        return nn.LeakyReLU(0.1,inplace=True)
    elif act_type=='silu':
        #sigmoid(x) = 1/(1 + e^(-x))
        #f(x) = x * sigmoid(x)
        return nn.SiLU()
    elif act_type=='mish':
        return nn.Mish()
    elif act_type is not None:
        return nn.Identity()
def get_conv(c1,c2,k,s,p,d,g,bias):
    #bias是一个布尔值，控制是否在卷积操作后添加偏置项，添加偏置项的卷积有更强的表达能力，如果有归一化层就不用了
    #dilation为空洞卷积，值为1时为标准卷积，大于1为空洞卷积，其感受域扩大
    return nn.Conv2d(c1,c2,k,stride=s,padding=p,dilation=d,groups=g,bias=bias)
class Conv(nn.Module):
    def __init__(self,c1,c2,k,s=1,p=0,d=1,norm_type='BN',act_type='relu',depthwise=False):
        super().__init__()
        conv = []
        add_bias = False if norm_type is not None else True
        #构建depthwise + pointwise卷积：即深度卷积和逐点卷积
        if depthwise:
            #depthwise卷积
            conv.append(get_conv(c1,c1,k=k,s=s,p=p,d=d,g=c1,bias=add_bias))
            if norm_type:
                conv.append(get_norm(norm_type,c1))
            if act_type:
                conv.append(get_act(act_type))
            #pointwise卷积
            conv.append(get_conv(c1,c2,k=1,s=1,p=0,d=d,g=1,bias=add_bias))
            if norm_type:
                conv.append(get_norm(norm_type,c2))
            if act_type:
                conv.append(get_act(act_type))
        else:
            #标准卷积
            conv.append(get_conv(c1,c2,k=k,s=s,p=p,d=d,g=1,bias=add_bias))
            if norm_type:
                conv.append(get_norm(norm_type,c2))
            if act_type:
                conv.append(get_act(act_type))
        self.out = nn.Sequential(*conv)
    def forward(self,x):
        return self.out(x)
