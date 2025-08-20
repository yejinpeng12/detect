from torch import nn
from Conv import Conv
import torch

class Bottleneck(nn.Module):
    def __init__(self,in_c,out_c,shortcut=True,g=1,k=(3,3),e=0.5):
        #shortcut是确定是否使用残差，True带残差
        super().__init__()
        c_ = int(out_c * e)
        self.conv1 = Conv(in_c,c_,k[0],1)
        self.conv2 = Conv(c_,out_c,k[1],1,g=g)
        self.add = shortcut and in_c == out_c
    def forward(self,x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Module):
    def __init__(self,in_c,out_c,n=1,shortcut=False,g=1,e=0.5):
        super().__init__()
        self.c_ = int(out_c * e)
        self.conv1 = Conv(in_c, 2 * self.c_,1,1)
        self.conv2 = Conv((2 + n) * self.c_, out_c,1)
        self.m = nn.ModuleList(Bottleneck(self.c_,self.c_,shortcut,g,k=((3,3),(3,3)),e=1.0) for _ in range(n))
    def forward(self,x):
        y  = list(self.conv1(x).chunk(2,1))#chunk把self.conv1输出的结果的通道分为数量相同的两部分，y变为[y1,y2]
        y.extend(m(y[-1]) for m in self.m)#不断对列表的最后一个元素进行计算，向y加入新的值，y变为[y1,y2,y3,y4,y5,y6,...,yn+2]
        return self.conv2(torch.cat(y,dim=1))#在通道数融合

class SPPF(nn.Module):
    def __init__(self,in_c,out_c,k=5):
        super().__init__()
        c_ = in_c // 2
        self.conv1 = Conv(in_c, c_,1,1)
        self.conv2 = Conv(c_ * 4,out_c,1,1)
        self.m = nn.MaxPool2d(kernel_size=k,stride=1,padding=k // 2)
    def forward(self,x):
        y = [self.conv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y,dim=1))

