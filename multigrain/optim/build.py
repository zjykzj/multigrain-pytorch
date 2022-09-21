# -*- coding: utf-8 -*-

"""
@date: 2022/9/21 上午11:52
@file: build.py
@author: zj
@description: 
"""

from torch.optim import SGD


def build_optim(args, model):
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    return optimizer
