from torch import nn
# 残差连接（Residual Connection）和层归一化
# 通过堆叠多层网络，可以提高网络的表示能力，但也容易引入梯度消失或梯度爆炸等问题
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))