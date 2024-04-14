import torch
from torch import nn


class Encoder(nn.Module):
    """基础编码器接口"""

    def __init__(self, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


def init_seq2seq(module):
    """初始化Seq2Seq模型"""

    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])  # type: ignore


class Seq2SeqEncoder(Encoder):
    """RNN编码器用于Seq2Seq学习"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        # X的形状是(batch_size, num_steps)
        embs = self.embedding(X.type(torch.int64).T)  # type: ignore
        # embs的形状是(batch_size, num_steps, embed_size)
        output, state = self.rnn(embs)
        # output的形状是(num_steps, batch_size, num_hiddens)
        # state的形状是(num_layers, batch_size, num_hiddens)
        return output, state
