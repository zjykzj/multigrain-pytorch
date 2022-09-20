# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils.data.sampler import BatchSampler
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence

"""
采样器仅执行采样操作

RA的目的在于扩充采样数据，通过重复采样的方式来提高采样数量

有两个关键参数

1. repetition： 单张图片扩充的次数（注意：扩充的图片是连续输出的，也就是说，单批次图片中会出现重复图片）
2. len_factor：扩充因子，针对整个数据集的扩充倍数（注意：基于扩充因子得到的单轮数据集长度是一致的，而repetition的使用会导致某些图片在该轮训练中并未得到使用）

另外还包括常规的操作：

1. batch_size： 单批次大小
2. shuffle：是否打乱训练
3. drop_last： 是否舍弃最后不满足batch_size长度的图片列表
"""


class RASampler(torch.utils.data.Sampler):
    """
    Batch Sampler with Repeated Augmentations (RA)
    - dataset_len: original length of the dataset
    - batch_size
    - repetitions: instances per image
    - len_factor: multiplicative factor for epoch size
    """

    def __init__(self, dataset_len, batch_size, repetitions=1, len_factor=1.0, shuffle=False, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def shuffler(self):
        if self.shuffle:
            # 生成随机下标
            new_perm = lambda: iter(np.random.permutation(self.dataset_len))
        else:
            # 生成顺序下标
            new_perm = lambda: iter(np.arange(self.dataset_len))
        # 创建可迭代对象
        shuffle = new_perm()
        # 创建迭代器
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            # 重复次数
            for repetition in range(self.repetitions):
                yield index

    def __iter__(self):
        """
        迭代器
        :return:
        """
        shuffle = iter(self.shuffler())
        seen = 0
        batch = []
        for _ in range(self.len_images):
            index = next(shuffle)
            batch.append(index)
            if len(batch) == self.batch_size:
                # 批量加载
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # 批量采样长度
        # 如果不舍弃drop_last，那么
        if self.drop_last:
            return self.len_images // self.batch_size
        else:
            return (self.len_images + self.batch_size - 1) // self.batch_size


def list_collate(batch):
    """
    存在输入大小变化的情况，所以返回列表而不是张量
    Collate into a list instead of a tensor to deal with variable-sized inputs
    """
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy':
        if elem_type.__name__ == 'ndarray':
            return list_collate([torch.from_numpy(b) for b in batch])
    elif isinstance(batch[0], Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    return default_collate(batch)
