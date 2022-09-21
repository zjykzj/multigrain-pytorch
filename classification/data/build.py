# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:25
@file: build.py
@author: zj
@description: 
"""

import numpy as np

from torchvision import transforms

from .lighting import Lighting
from .bound import Bound
from .resize import Resize

"""
从实现上看，不管是训练还是测试阶段，预处理器都进行了固定大小缩放操作，保证同一批图片都是相同大小的
"""


class Empty(object):
    def __call__(self, data):
        return data


def build_transform(input_size, eig_val=None, eig_vec=None, mean=None, std=None):
    """
    MultiGrain在使用过程中基于ImageNet的均值/方差以及特征值和特征向量进行了数据预处理

    当前仍旧使用ImageNet数据集，所以仍旧使用这个配置
    :param input_size:
    :param eig_val:
    :param eig_vec:
    :param mean:
    :param std:
    :return:
    """

    if eig_val is None or eig_vec is None:
        eig_val = [0.2175, 0.0188, 0.0045]
        eig_vec = np.array([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]
        ])

    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        Empty() if eig_val is None or eig_vec is None else Lighting(0.1, eig_val, eig_vec),
        Bound(0., 1.),
        Empty() if mean is None or std is None else transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        Resize(input_size, largest=True),  # to maintain same ratio w.r.t. 224 images
        transforms.ToTensor(),
        Empty() if mean is None or std is None else transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform
