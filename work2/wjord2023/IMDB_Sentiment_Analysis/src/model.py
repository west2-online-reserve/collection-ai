from torch import nn


# 构建模型
class IMDBModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(IMDBModel, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 双向LSTM层
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 2是因为是双向的

        # Dropout层
        self.dropout = nn.Dropout(0.7)

    def forward(self, input_seq):
        # 词嵌入
        embedded_seq = self.embedding(input_seq)

        # 双向LSTM编码
        lstm_output, _ = self.bilstm(embedded_seq)

        # 获取最后一个时间步的输出
        lstm_output = lstm_output[:, -1, :]

        # 使用Dropout层防止过拟合
        lstm_output = self.dropout(lstm_output)

        # 通过全连接层进行分类
        output = self.fc(lstm_output)

        return output
