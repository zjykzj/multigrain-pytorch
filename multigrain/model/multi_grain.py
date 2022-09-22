# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午8:09
@file: multigrain.py
@author: zj
@description: 
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torchvision.models import resnet

from criterion.build import KEY_OUTPUT, KEY_FEAT
from criterion.build import KEY_ANCHOR, KEY_POSITIVE, KEY_NEGATIVE

from .distance_weighted_sampler import DistanceWeightedSampling


def l2n(x, eps=1e-6, dim=1):
    """
    基于指定维度计算L2范数，执行归一化操作
    """
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x


def gem(x, p=3.0, eps=1e-6):
    x = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    return x


class MultiGrain(Module):

    def __init__(self, backbone, p=3.0) -> None:
        super().__init__()

        if backbone == 'resnet50':
            model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2)
            children = list(model.named_children())

            self.features = nn.Sequential(OrderedDict(children[:-2]))
            # self.pool = children[-2][1]
            self.classifier = children[-1][1]
        else:
            raise ValueError(f'{backbone} does not supports')

        self.p = p
        self.pool = gem
        self.normalize = l2n
        self.weighted_sampling = DistanceWeightedSampling()

    def forward(self, x, target):
        features = self.features(x)
        embedding = self.pool(features, p=self.p)
        embedding = embedding.view(embedding.size(0), -1)
        classifier_output = self.classifier(embedding)

        normalized_embedding = self.normalize(embedding)
        res_dict = {
            KEY_FEAT: normalized_embedding,
            KEY_OUTPUT: classifier_output
        }

        weighted_embedding_dict = self.weighted_sampling.forward(normalized_embedding, target)
        res_dict.update(weighted_embedding_dict)
        return res_dict


if __name__ == '__main__':
    model = MultiGrain('resnet50')
    print(model)
