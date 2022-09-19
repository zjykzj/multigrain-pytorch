# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:52
@file: utils.py
@author: zj
@description: 
"""

import os


def ifmakedirs(path):
    # 创建文件夹
    if not os.path.exists(path):
        os.makedirs(path)
