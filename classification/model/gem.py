# -*- coding: utf-8 -*-

"""
@date: 2022/9/20 上午10:03
@file: gem.py
@author: zj
@description: 
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def add_bias_channel(x, dim=1):
    """
    增加偏置通道，原先大小为[N, K], 赋值成[N, 1]大小，初始化为1
    最后按照dim=1维度进行连接，得到[N, K+1]
    """
    one_size = list(x.size())
    one_size[dim] = 1
    one = x.new_ones(one_size)
    return torch.cat((x, one), dim)


def flatten(x, keepdims=False):
    """
    [B, C, H, W] -> [B, C*H*W]
    如果保持原先的维度，那么设置为[1, 1, B, C*H*W]
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y


class GeM(nn.Module):

    def __init__(self, p=3.0, learn_p=False, eps=1e-6, clamp=True, add_bias=False, keepdims=False) -> None:
        super(GeM, self).__init__()
        if not torch.is_tensor(p):
            p = torch.tensor(p)
            if learn_p:
                p.requires_grad = True
        self.p = p
        self.eps = eps
        self.clamp = clamp
        self.add_bias = add_bias
        self.keepdims = keepdims

    def forward(self, x):
        if self.p == math.inf or self.p is 'inf':
            x = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        elif self.p == 1 and not (torch.is_tensor(self.p) and self.p.requires_grad):
            x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        else:
            if self.clamp:
                x = x.clamp(min=self.eps)
            x = F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        if self.add_bias:
            x = add_bias_channel(x)
        if not self.keepdims:
            x = flatten(x)
        return x
