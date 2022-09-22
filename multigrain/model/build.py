# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午8:07
@file: build.py
@author: zj
@description:

将完整的分类网络拆分为u几个模块：

1. 主干Backbone
2. 池化层
3. 分类器

先不管白化操作，替换掉池化层，使用GeM池化

先创建自定义的分类模型，然后加入到MultiGrain架构

返回两部分内容：

1. 特征向量
2. 分类输出

"""

import torch

from .multi_grain import MultiGrain


def build_model(args):
    model = MultiGrain(args.backbone, p=args.pooling_exponent)

    return model
