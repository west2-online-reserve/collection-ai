import torch
import math
from torch import nn


def sequence_mask(X, valid_len, value=0.0):
    """在序列中屏蔽不相关的项"""
    """X: 3D tensor, valid_len: 1D tensor"""
    maxlen = X.size(1)  # maxlen表示序列的最大长度
    # mask为bool类型的tensor，shape为(1, maxlen)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    # 将X中mask为False的元素全部赋值为value
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    """X: 3D tensor, valid_lens: 1D or nD tensor"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果valid_lens是一个1D tensor，那么将它重复到与X的shape相同（shape[1]是序列长度）
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 否则将其转化为1D tensor
            valid_lens = valid_lens.reshape(-1)
        # 在最后一个轴上，X的不相关项被设置为一个很大的负值，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # W_k相当于一个shape为(key_size, num_hiddens)的矩阵
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # W_q相当于一个shape为(query_size, num_hiddens)的矩阵
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # W_v相当于一个shape为(num_hiddens, 1)的矩阵
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # valid_lens限制多少对键值对的注意力
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hiddens`)
        # `key` 的形状：(`batch_size`, 1, 键值对的个数, `num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # features的形状：(`batch_size`, 查询的个数, 键值对的个数, `num_hiddens`)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，键值对的个数)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，键值对的个数，值的维度)
        # bmm: batch matrix multiplication, 将注意力权重与values相乘
        # dropout: 丢弃一些注意力权重（变为0）, 保持形状不变
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[
            -1
        ]  # d是queries的最后一个维度的大小, 即查询的个数(q和k的长度相同)
        # 点积，scores的形状：(batch_size，查询的个数，键值对的个数)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 返回值的shape：(batch_size，查询的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多头注意力的并行计算,将queries, keys, values的形状转换"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # `queries`, `keys`, `values` shape：(batch_size, 查询的个数或键值对的个数, num_hiddens)
        # `valid_lens` shape: (batch_size, ) or (batch_size, 查询的个数)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # 变化后，qkv的形状是(batch_size*num_heads, 查询的个数或键值对的个数, num_hiddens/num_heads)
        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )
        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
