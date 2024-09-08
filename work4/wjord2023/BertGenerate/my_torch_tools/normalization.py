from torch import nn


class AddNorm(nn.Module):
    """AddNorm层"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # Y的形状和X一样， X是残差连接, Y是层的输出
        # 返回值的形状和X一样
        return self.ln(self.dropout(Y) + X)
