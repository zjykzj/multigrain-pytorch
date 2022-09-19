# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:26
@file: bound.py
@author: zj
@description: 
"""


class Bound(object):
    """
    设置数值最大、最小值，截断精度
    """

    def __init__(self, lower=0., upper=1.):
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        return data.clamp_(self.lower, self.upper)
