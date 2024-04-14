import torch
from .attention import AdditiveAttention
from torch import nn


class Decoder(nn.Module):
    """基础解码器接口"""

    def __init__(self, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class AttentionDecoder(Decoder):
    """带有注意力机制的解码器"""

    def __init__(self, **kwargs) -> None:
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带有Bahdanau注意力的循环神经网络解码器"""

    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ) -> None:
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 使用加性注意力，为了可以学习注意力权重
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):  # type: ignore
        # enc_valid_lens，告诉解码器哪一些是被padding填充的
        outputs, hidden_state = enc_outputs
        # 把batch_size放在第0维
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状是(batch_size, num_steps, num_hiddens)
        # hidden_state的形状是(num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)  # 再把num_steps放在第0维
        # X的形状是(num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状是(batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # key, value的形状是(batch_size, num_steps, num_hiddens),等于enc_outputs
            # context的形状是(batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 将嵌入后的输入和上下文连接起来
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，输出形状是(num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
