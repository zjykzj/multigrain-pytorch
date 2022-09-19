# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午8:29
@file: resnet.py
@author: zj
@description: 
"""
from typing import Type, Union, List, Optional, Callable

import torch
from torch import Tensor
from torch import nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(resnet.ResNet):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)

    def forward_backbone(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_classify(self, x):
        x = self.fc(x)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
