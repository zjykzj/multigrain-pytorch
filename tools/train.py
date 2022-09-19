# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:21
@file: train.py.py
@author: zj
@description: 
"""

from multigrain.data.build import build_data


def main(args):
    train_loader, val_loader = build_data(args)



if __name__ == '__main__':
    args = None
    main()
