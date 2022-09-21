# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:56
@file: build.py
@author: zj
@description: 
"""

from .imagenet import ImageNet


def build_dataset(data_root, transform, split: str = "train"):
    dataset = ImageNet(data_root, transform=transform, split=split)
    return dataset
