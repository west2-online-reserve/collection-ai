import torch.nn as nn

from src.model.embedding.embedding import MyBERTEmbedding
from src.model.transformer import TransformerBlock


class MyChineseBERT(nn.Module):

    def __init__(self, vocab_size, hidden=256, n_layers=12, attn_heads=8, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = MyBERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.linear_next_sentence = nn.Linear(hidden, 2)
        self.linear_mask_lm = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x1 = self.softmax(self.linear_next_sentence(x[:, 0]))
        x2 = self.softmax(self.linear_mask_lm(x))
        return x1, x2