# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午8:07
@file: build.py
@author: zj
@description:

将完整的分类网络拆分为u几个模块：

1. 主干Backbone
2. 池化层
3. 分类器

先不管白化操作，替换掉池化层，使用GeM池化

"""


def build_model():
