import torch
from torch import nn, optim

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        """
        super(RNN, self).__init__()

        # [0-10001] => [100]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # [100] => [256]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: [seq_len, b] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 100]
        embedding = self.dropout(self.embedding(x))

        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.rnn(embedding)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out