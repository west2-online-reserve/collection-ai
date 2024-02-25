from torch import nn

from src.model.attention.mulitihead_attention import MultiHeadedAttention
from src.model.utils.positionwise_feedforward import PositionwiseFeedForward
from src.model.utils.sublayer_connection import SublayerConnection


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # 表示注意机制的 lambda 函数。注意机制应用于输入序列 _x 三次（查询、键和值相同），并且使用 mask 来屏蔽注意分数中的特定位置。
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
