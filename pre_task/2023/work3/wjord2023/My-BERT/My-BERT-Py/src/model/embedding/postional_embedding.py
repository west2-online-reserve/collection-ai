import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 初始化位置编码矩阵，大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False  # 设置位置编码不需要梯度更新

        # 生成位置信息，表示为从 0 到 max_len-1 的浮点数，形状为 (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 计算位置编码的除数项，形状为 (d_model/2,)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 使用 sin 和 cos 函数计算位置编码矩阵中的值，分别作用于偶数和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码矩阵添加一个维度，并注册为模型的缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        输入 x 的形状为 (batch_size, sequence_length, d_model)，
        输出的位置编码将选择适当长度的部分返回。
        """
        return self.pe[:, :x.size(1)]
