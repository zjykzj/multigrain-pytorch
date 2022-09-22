# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:52
@file: imagenet.py
@author: zj
@description: 
"""
from typing import Any, Tuple

import numpy as np

from torchvision.datasets import imagenet


class ImageNet(imagenet.ImageNet):

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

