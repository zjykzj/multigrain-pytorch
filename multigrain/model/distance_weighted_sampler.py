# -*- coding: utf-8 -*-

"""
@date: 2022/9/22 上午10:02
@file: distance_weighted_sampler.py
@author: zj
@description: 
"""

from torch import nn
import torch
import numpy as np

from criterion.build import KEY_ANCHOR, KEY_POSITIVE, KEY_NEGATIVE


class DistanceWeightedSampling(nn.Module):
    r"""Distance weighted sampling.
    See "sampling matters in deep embedding learning" paper for details.
    Implementation similar to https://github.com/chaoyuaw/sampling_matters
    """

    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4):
        super().__init__()
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    @staticmethod
    def get_distance(x):
        """
        Helper function for margin-based loss. Return a distance matrix given a matrix.
        Returns 1 on the diagonal (prevents numerical errors)
        """
        # 获取特征列表长度
        n = x.size(0)
        # 计算每个特征向量的平方和 [N, D] -> [N, 1]
        square = torch.sum(x ** 2.0, dim=1, keepdim=True)
        # 计算向量之间的欧式距离
        # square + square.t(): [N, 1] + [1, N] -> [N, N]
        # torch.matmul(x, x.t()): [N, D] * [D, N] -> [N, N]
        # 等价于 x1**2 + x2**2 - 2*x1*x2 = (x1 - x2)**2
        # 这种情况下，矩阵对角特征值为0，因为x1 == x2
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        # 开根号，计算欧式距离
        # 对角特征设置为1, 避免数值计算错误
        return torch.sqrt(distance_square + torch.eye(n, dtype=x.dtype, device=x.device))

    def forward(self, embedding, target):
        """
        embedding: [N, D]
        target: [N]

        Inputs:
            - embedding: embeddings of images in batch
            - target: id of instance targets

        Outputs:
            - a dict with
               * 'anchor_embeddings'
               * 'negative_embeddings'
               * 'positive_embeddings'
               with sampled embeddings corresponding to anchors, negatives, positives
        """

        # 获取特征向量的个数和维度
        B, C = embedding.size()[:2]
        embedding = embedding.view(B, C)

        # 计算特征向量之间的欧式距离
        distance = self.get_distance(embedding)
        # 精度截断
        distance = torch.clamp(distance, min=self.cutoff)

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(C)) * torch.log(distance)
                       - (float(C - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))
        # 计算指数权重
        weights = torch.exp(log_weights - log_weights.max())

        # [N] -> [1, N]
        unequal = target.view(-1, 1)
        # 计算不相同target的下标
        unequal = (unequal != unequal.t())

        # 计算掩码，过滤对角特征以及距离小于指定阈值的特征
        weights = weights * (unequal & (distance < self.nonzero_loss_cutoff)).float()
        # 剩余特征进行归一化
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.detach().cpu().numpy()
        unequal_np = unequal.cpu().numpy()

        # 遍历每条特征
        for i in range(B):
            # 计算和特征i拥有相同标签的特征的下标
            # 这里面包括了下标i
            same = (1 - unequal_np[i]).nonzero()[0]

            # 如果权重求和不为空
            # 注意： 仅不相同标签的特征下标才会出现权重值
            if np.isnan(np_weights[i].sum()):  # 0 samples within cutoff, sample uniformly
                np_weights_ = unequal_np[i].astype(float)
                # 权重归一化
                np_weights_ /= np_weights_.sum()
            else:
                # 上面已经归一化了
                np_weights_ = np_weights[i]

            # 采样负样本列表，长度和正样本列表一致
            try:
                # 基于采样概率进行，对于概率为0的情况，其采集可能性为0
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_, replace=False).tolist()
            except ValueError:  # cannot always sample without replacement
                # 如果负样本长度小于正样本长度，那么允许重复采样
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_).tolist()

            for j in same:
                if j != i:
                    # 采集正样本对
                    a_indices.append(i)
                    p_indices.append(j)

        # return {'anchor_embeddings': embedding[a_indices],
        #         'negative_embeddings': embedding[n_indices],
        #         'positive_embeddings': embedding[p_indices]}
        return {KEY_ANCHOR: embedding[a_indices],
                KEY_POSITIVE: embedding[n_indices],
                KEY_NEGATIVE: embedding[p_indices]}
