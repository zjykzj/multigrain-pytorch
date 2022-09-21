# -*- coding: utf-8 -*-

"""
@date: 2022/9/19 下午7:21
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader

from transform.build import build_transform
from dataset.build import build_dataset
from sampler.ra_sampler import RASampler


def build_data(args):
    train_transform, val_transform = build_transform(args.input_size)

    train_dataset = build_dataset(args.data_root, transform=train_transform, split='train')
    val_dataset = build_dataset(args.data_root, transform=val_transform, split='val')

    train_sampler = RASampler(len(train_dataset), args.batch_size,
                              repetitions=args.repeated_augmentations,
                              len_factor=args.epoch_len_factor, shuffle=True, drop_last=False)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle_val,
                            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader
