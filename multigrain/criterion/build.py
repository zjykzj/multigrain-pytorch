# -*- coding: utf-8 -*-

"""
@date: 2022/9/20 上午11:02
@file: build.py
@author: zj
@description: 
"""

import torch

from .margin import MarginLoss


def build_criterion(args):
    cross_entropy = torch.nn.CrossEntropyLoss()

    margin = MarginLoss(args.beta_init)

    # 创建组合损失函数
    criterion = MultiCriterion(dict(cross_entropy=cross_entropy_criterion, margin=margin_criterion),
                               skip_zeros=(args.repeated_augmentations == 1))
    return criterion
