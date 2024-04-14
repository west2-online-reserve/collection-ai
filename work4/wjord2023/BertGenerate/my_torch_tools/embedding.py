import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        # num_hiddens: 词向量的维度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的`P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        # 计算位置编码的公式的实现
        X = torch.arange(0, max_len).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # X的形状是(batch_size, num_steps, num_hiddens)
        # P的形状是(1, num_steps, num_hiddens) 对p的第一个维度进行广播
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)
