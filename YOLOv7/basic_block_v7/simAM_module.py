import torch
import torch.nn as nn

class simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f') % self.e_lambda
        return s

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)#计算每个位置和均值的平方差
        #计算注意力权重，这里是能量函数
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)#使用sigmoid将注意力权重映射到0~1,返回注意力权重与输入的乘积
        #高能量区的权重较高，会受到关注，低能量区受到的关注会较少