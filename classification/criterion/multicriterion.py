# -*- coding: utf-8 -*-

"""
@date: 2022/9/20 上午11:04
@file: multicriterion.py
@author: zj
@description: 
"""
from torch import nn
from collections import OrderedDict as OD


"""
多损失融合 对于MultiGrain而言，进行分类任务和检索任务的训练


"""


class MultiCriterion(nn.Module):
    """
    多损失加权

    每个损失函数对应一个加权因子，以及可能会有多个输入（通过input_key判定）
    1. cross_entropy=cross_entropy_criterion

        cross_entropy_criterion = (cross_entropy,
                               ('classifier_output', 'classifier_target'),
                               args.classif_weight)

    2. margin=margin_criterion

        margin_criterion = (margin,
                    ('anchor_embeddings', 'negative_embeddings', 'positive_embeddings'),
                    1.0 - args.classif_weight)

    3. skip_zeros=(args.repeated_augmentations == 1)

    Holds a dict of multiple losses with a weighting factor for each loss.
    - losses_dict: should be a dict with name as key and (loss, input_keys, weight) as values.
    - skip_zero: skip the computation of losses with 0 weight
    """

    def __init__(self, losses_dict, skip_zeros=False):
        super().__init__()
        self.losses = OD()
        self.input_keys = OD()
        self.weights = OD()
        for name, (loss, input_keys, weight) in losses_dict.items():
            # weight表示加权因子，用于控制不同损失之间的训练权重
            self.losses[name] = loss
            self.input_keys[name] = input_keys
            self.weights[name] = weight
        self.losses = nn.ModuleDict(self.losses)
        self.skip_zeros = skip_zeros

    def forward(self, input_dict):
        return_dict = {}
        loss = 0.0
        for name, module in self.losses.items():
            # 分别计算不同损失
            for k in self.input_keys[name]:
                # 判断该损失所需的输入是否全部存在
                if k not in input_dict:
                    raise ValueError('Element {} not found in input.'.format(k))
            if self.weights[name] == 0.0 and self.skip_zeros:
                continue
            # 计算损失，输入损失需要的值
            # 对于分类损失CrossEntropyLoss而言，需要输入
            # classifier_output: 分类层输出，大小为[N, N_classes]
            # classifier_target： 图像对应标签，大小为[N]
            # 对于检索u损失MarginLoss而言，需要输入
            # anchor_embeddings
            # negative_embeddings
            # positive_embeddings
            this_loss = module(*[input_dict[k] for k in self.input_keys[name]])
            return_dict[name] = this_loss
            # 加权计算
            loss = loss + self.weights[name] * this_loss
        # 最后总的损失
        return_dict['loss'] = loss
        return return_dict
