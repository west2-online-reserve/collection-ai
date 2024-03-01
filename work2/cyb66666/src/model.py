import torch
import pickle
import torch.nn as nn

max_len = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_value = []
acc_rate = []


class imdb_model(nn.Module):
    def __init__(self,w2s):
        super().__init__()
        self.embedding_dims = 300
        self.hidden_size = 100
        self.num_layer = 2
        self.bidirectional = True
        self.num_direction = 2 if self.bidirectional else 1
        self.max_len = max_len
        self.embedding = nn.Embedding(len(w2s) + 1, embedding_dim=self.embedding_dims)
        self.drop_out = 0.2
        self.Dropout = nn.Dropout(self.drop_out)
        self.lstm = nn.LSTM(input_size=self.embedding_dims, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, batch_first=True, bidirectional=False)
        self.li = nn.Linear(self.hidden_size * 1, 2)

    def forward(self, input):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        output = self.embedding(input)
        output, (hn, cn) = self.lstm(output)
        output = output[:, -1, :]
        # output = torch.mean(output, dim=1)
        output = torch.sigmoid(self.li(output))
        return output
