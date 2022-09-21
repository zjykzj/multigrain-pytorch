# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:52
@file: imagenet.py
@author: zj
@description: 
"""
from typing import Any

import numpy as np

from torchvision.datasets import imagenet


class ImageNet(imagenet.ImageNet):

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)

    def parse_archives(self) -> None:
        super().parse_archives()

    @property
    def split_folder(self) -> str:
        return super().split_folder()

    def extra_repr(self) -> str:
        return super().extra_repr()
