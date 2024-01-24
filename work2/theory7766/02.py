import torch
from torch import nn
import tool

class GRUNet(nn.Module):
    def __init__(self,hidden):
        super(GRUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(64*4*4, hidden, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm5 = nn.BatchNorm1d(hidden)
        self.fc = nn.Linear(hidden, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1, 64*4*4)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.batch_norm5(out)
        out = self.fc(out)
        return out


def main():
    net = GRUNet(hidden = 512)
    #net.load_state_dict(torch.load('CIFAR100_gru__7.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loss = nn.CrossEntropyLoss()
    #updater = torch.optim.Adam(net.parameters(),lr=0.00005)
    updater = torch.optim.Adam(net.parameters(), lr=0.001)
    # 读取数据集
    batch_size = 64
    train_iter, test_iter = tool.load_data_cifar(batch_size)
    #
    # for i in range(batch_size-1):
    #     examples = enumerate(train_iter)
    #     batch_idx,(imgs,labels) = next(examples)
    #     print(labels)
    # 开始训练
    epoch_num = 15
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = tool.train(net, epoch_num, loss, updater,
                                                                                train_iter, test_iter, device)
    torch.save(net.state_dict(), 'CIFAR100_gru__7.pth')
    tool.show_acc(train_acc_list, test_acc_list)
    tool.show_loss(train_loss_list, test_loss_list)

if __name__ == '__main__':
    main()
