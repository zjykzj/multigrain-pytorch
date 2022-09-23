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

KEY_ANCHOR = 'anchor'
KEY_POSITIVE = 'positive'
KEY_NEGATIVE = 'negative'

from .cross_entropy_loss import CrossEntropyLoss
from .margin_loss import MarginLoss
from .multi_criterion import MultiCriterion


def build_criterion(args):
    """
    对于分类任务而言，使用CrossEntropyLoss
    :param args:
    :return:
    """
    cross_entropy_loss = CrossEntropyLoss(label_smoothing=args.label_smoothing)
    margin_loss = MarginLoss(beta_init=args.beta_init)

    multi_criterion = MultiCriterion(classify_loss=cross_entropy_loss, retrieval_loss=margin_loss,
                                     classify_weight=args.classify_weight)

    return multi_criterion, margin_loss, cross_entropy_loss
