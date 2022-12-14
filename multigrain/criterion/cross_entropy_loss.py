# -*- coding: utf-8 -*-

"""
@date: 2022/9/21 上午11:40
@file: cross_entropy_loss.py
@author: zj
@description:

对于分类任务，需要输出向量outputs: [N, N_class]以及每个图像对应标签targets: [N]

"""
from typing import Dict

from torch import nn

from .build import KEY_OUTPUT, KEY_FEAT


class CrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing=None):
        super(CrossEntropyLoss, self).__init__()
        if label_smoothing is None or label_smoothing == 0.0:
            self.loss = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smoothing)

    def forward(self, input_dict: Dict, targets):
        assert KEY_OUTPUT in input_dict.keys()

        outputs = input_dict[KEY_OUTPUT]
        assert len(outputs) == len(targets)

        loss = self.loss(outputs, targets)
        return loss
