import torch
from torch import nn
import tool
from torchvision.models import resnet

class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net = resnet.ResNet(resnet.BasicBlock,[2,2,2,2],num_classes=100)
    net = net1()
    net.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 读取数据集
    batch_size = 64
    train_iter, test_iter = tool.load_data_cifar(batch_size)
    # 开始训练
    epoch_num = 100
    tool.train(net, epoch_num, loss, updater, train_iter, test_iter, device)


if __name__ == '__main__':
    main()
