# -*- coding: utf-8 -*-

"""
@date: 2022/9/22 上午10:39
@file: multi_criterion.py
@author: zj
@description: 
"""

from torch import nn

from .build import KEY_ANCHOR, KEY_POSITIVE, KEY_NEGATIVE


class MultiCriterion(nn.Module):

    def __init__(self, classify_loss=None, retrieval_loss=None) -> None:
        super().__init__()
        assert classify_loss is not None and retrieval_loss is not None

        self.classify_loss = classify_loss
        self.retrieval_loss = retrieval_loss

    def forward(self, input_dict, target):
        loss1 = self.classify_loss(input_dict, target)

        anchors = input_dict[KEY_ANCHOR]
        positives = input_dict[KEY_POSITIVE]
        negatives = input_dict[KEY_NEGATIVE]

        loss2 = self.retrieval_loss(anchors, positives, negatives)

        return loss1 + loss2, loss1, loss2
