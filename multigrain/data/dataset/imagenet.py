# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:52
@file: imagenet.py
@author: zj
@description: 
"""
from typing import Any

import numpy as np

from torchvision.datasets import imagenet


class ImageNet(imagenet.ImageNet):
    # ImageNet类别数
    NUM_CLASSES = 1000
    # 根据训练集计算得到的均值和标准差
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # 计算RGB三通道的均值和方差，然后计算整个数据集的协方差矩阵, 得到各个通道的特征值和特征向量
    # RGB三通道的特征值和特征向量
    # 适用于PCA抖动预处理
    EIG_VALS = [0.2175, 0.0188, 0.0045]
    EIG_VECS = np.array([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ])

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)

    def parse_archives(self) -> None:
        super().parse_archives()

    @property
    def split_folder(self) -> str:
        return super().split_folder()

    def extra_repr(self) -> str:
        return super().extra_repr()
