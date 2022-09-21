# -*- coding: utf-8 -*-

"""
@date: 2022/9/20 上午11:02
@file: build.py
@author: zj
@description: 
"""

import torch

from .margin import MarginLoss

KEY_OUTPUT = 'output'
KEY_TARGET = 'target'
KEY_FEAT = 'feat'


def build_criterion(args):
    """
    对于分类任务而言，使用CrossEntropyLoss
    :param args:
    :return:
    """
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    return cross_entropy_loss
