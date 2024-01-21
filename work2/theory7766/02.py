import torch
from torch import nn
import tool

class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(3072, 256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # 添加 dropout 层
        self.batch_norm = nn.BatchNorm1d(256)  # 添加批标准化层
        self.fc = nn.Linear(256, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1, 3072)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)  # 应用 dropout
        out = self.batch_norm(out)  # 应用批标准化
        out = self.fc(out)
        return out

def main():
    net = GRUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.Adam(net.parameters(), lr=0.001)
    # 读取数据集
    batch_size = 128
    train_iter, test_iter = tool.load_data_cifar(batch_size)
    # 开始训练
    epoch_num=10
    tool.train(net, epoch_num, loss, updater, train_iter, test_iter,device)



if __name__=='__main__':
    main()