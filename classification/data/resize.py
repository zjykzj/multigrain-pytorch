# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:24
@file: resize.py
@author: zj
@description: 
"""

import torchvision.transforms.functional as F
from torchvision import transforms


class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping

    图像缩放，较大边长缩放到指定size大小
    """

    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if (h < w) == largest:
            # 如果宽大于高，那么宽设置为size，高按照等比例缩放
            w, h = size, int(size * h / w)
        else:
            # 如果宽小于高，那么高设置为size，宽按照等比例缩放
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)
