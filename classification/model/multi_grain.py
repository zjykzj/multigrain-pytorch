# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午8:09
@file: multigrain.py
@author: zj
@description: 
"""

from collections import OrderedDict

from torch import nn
from torch.nn import Module
from torchvision.models import resnet

from criterion.build import KEY_OUTPUT, KEY_FEAT


class MultiGrain(Module):

    def __init__(self, backbone, ) -> None:
        super().__init__()

        if backbone == 'resnet50':
            model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2)
            children = list(model.named_children())

            self.features = nn.Sequential(OrderedDict(children[:-2]))
            self.pool = children[-2][1]
            self.classifier = children[-1][1]
        else:
            raise ValueError(f'{backbone} does not supports')

    def forward(self, x):
        features = self.features(x)
        embedding = self.pool(features)
        embedding = embedding.view(embedding.size(0), -1)

        classifier_output = self.classifier(embedding)

        return {
            KEY_FEAT: embedding,
            KEY_OUTPUT: classifier_output
        }


if __name__ == '__main__':
    model = MultiGrain('resnet50')
    print(model)
