# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:24
@file: lighting.py
@author: zj
@description: 
"""

import torch


class Lighting(object):
    """
    PCA jitter transform on tensors

    See https://zhuanlan.zhihu.com/p/69439309
    PCA抖动：首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值，用来做PCA Jittering
    """

    def __init__(self, alpha_std, eig_val, eig_vec):
        self.alpha_std = alpha_std
        self.eig_val = torch.as_tensor(eig_val, dtype=torch.float).view(1, 3)
        self.eig_vec = torch.as_tensor(eig_vec, dtype=torch.float)

    def __call__(self, data):
        if self.alpha_std == 0:
            return data
        alpha = torch.empty(1, 3).normal_(0, self.alpha_std)
        rgb = ((self.eig_vec * alpha) * self.eig_val).sum(1)
        data += rgb.view(3, 1, 1)
        data /= 1. + self.alpha_std
        return data
