from torch import nn


# 为输入的每个词生成对应的词嵌入（word embedding）。
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
