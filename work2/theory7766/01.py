import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import tool

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # 定义残差块里连续的2个卷积层
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv_block(x)
        out2 = self.shortcut(x) + out1
        out2 = nn.functional.relu(out2)
        return out2


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes):
        super(ResNet, self).__init__()
        # layer1时的channel
        self.in_channels = 64
        # 第一层单独的卷积层
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            # nn.Dropout(0.25)
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # 将图片像素强制转换为1*1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * 1 * 1, num_classes)
        self.dropout = nn.Dropout(0.3)

    # 构建重复残差块
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet(ResBlock, num_classes=20)
    #net.load_state_dict(torch.load('CIFAR100_resnet6.pth'))
    net.to(device)

    loss = nn.CrossEntropyLoss()
    loss.to(device)
    updater = torch.optim.Adam(net.parameters(), lr=0.0001)
    # 读取数据集
    batch_size = 64
    train_iter, test_iter = tool.load_data_cifar(batch_size)
    # 开始训练
    epoch_num = 5
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = tool.train(net, epoch_num, loss, updater,
                                                                           train_iter, test_iter,device)
    torch.save(net.state_dict(), 'CIFAR100_resnet.pth')
    tool.show_acc(train_acc_list, test_acc_list)
    tool.show_loss(train_loss_list, test_loss_list)

if __name__ == '__main__':
    main()
