# -*- coding: utf-8 -*-

"""
@date: 2022/9/22 上午10:37
@file: margin_loss.py
@author: zj
@description: 
"""

from torch import nn
import torch
import numpy as np


class MarginLoss(nn.Module):
    r"""Margin based loss.

    Parameters
    ----------
    beta_init: float
        Initial beta
    margin : float
        Margin between positive and negative pairs.
    """

    def __init__(self, beta_init=1.2, margin=0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self._margin = margin

    def forward(self, anchor_embeddings, negative_embeddings, positive_embeddings, eps=1e-8):
        """

        Inputs:
            - input_dict: 'anchor_embeddings', 'negative_embeddings', 'positive_embeddings'

        Outputs:
            - Loss.
        """

        # 计算正样本对之间的欧式距离
        d_ap = torch.sqrt(torch.sum((positive_embeddings - anchor_embeddings) ** 2, dim=1) + eps)
        # 计算负样本对之间的欧式距离
        d_an = torch.sqrt(torch.sum((negative_embeddings - anchor_embeddings) ** 2, dim=1) + eps)

        # 计算正样本对损失
        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        # 计算负样本对损失
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        # 计算符合条件的数目
        pair_cnt = float(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).item())

        # 归一化操作
        # Normalize based on the number of pairs
        loss = (torch.sum(pos_loss + neg_loss)) / max(pair_cnt, 1.0)

        return loss
