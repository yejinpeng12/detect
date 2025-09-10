from torch import nn

def autopad(k,p=None,d=1):#kernel, padding, dilation
    if d > 1:
        k = d * (d - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k ,int) else [x // 2 for x in k] #auto-pad
    return p
class Conv(nn.Module):
    def __init__(self,in_c,out_c,k=3,s=1,p=None,g=1,d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,k,s,autopad(k,p,d),groups=g,dilation=d,bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))