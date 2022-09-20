# -*- coding: utf-8 -*-

"""
@date: 2022/9/20 上午10:10
@file: model_base.py
@author: zj
@description:

拆分单个网络，进行

1. 主干网络的计算
2. 池话层的计算
3. 分类层的计算

"""

from abc import abstractmethod, ABCMeta

import torch.nn as nn


class ModelBase(nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward_backbone(self, x):
        pass

    @abstractmethod
    def forward_pool(self, x):
        pass

    @abstractmethod
    def forward_classify(self, x):
        pass
