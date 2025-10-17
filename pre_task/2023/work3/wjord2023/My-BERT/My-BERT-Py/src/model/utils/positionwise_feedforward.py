import torch.nn as nn
import torch.nn.functional as F


# 助于在编码器和解码器层中引入非线性特征映射，增加模型的表示能力。
class PositionwiseFeedForward(nn.Module):
    # d_ff:Feedforward 层的输出维度
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    # 新版torch加入了gelu，我们就不用再写gelu了
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
