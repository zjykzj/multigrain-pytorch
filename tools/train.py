# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:21
@file: train.py.py
@author: zj
@description: 

步骤：

1. 加载数据
    1. 创建预处理器
    2. 创建数据类
    3. 创建采样器
    4. 创建数据加载器

"""

import torch
from torch.optim import Optimizer

from multigrain.data.build import build_data
from multigrain.model.build import build_model
from multigrain.criterion.build import build_criterion
from multigrain.optim.build import build_optim


def train(dataloader, model, criterion, optimizer: Optimizer, device):
    model.train()

    for idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()
        output_dict = model(images)
        loss = criterion(output_dict, targets)

        loss.backward()
        optimizer.step()


@torch.no_grad()
def val(dataloader, model, criterion):
    model.eval()


def main(args):
    train_loader, val_loader = build_data(args)

    device = torch.device('cuda:0')
    model = build_model(args).to(device)

    criterion = build_criterion(args).to(device)
    optimizer = build_optim(args, model)


if __name__ == '__main__':
    args = None
    main()
