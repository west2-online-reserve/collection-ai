import random
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from IPython import display

# 加载CIFAR-100数据集
def load_data_cifar(batch_size):
    # ToTensor()将数值归一化到[0,1]
    # Normalize()使用均值和标准差对张量图片进行归一化,将数据分布到(-1,1)，均值为0，方差为1
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose( [transforms.ToTensor(),
                                 transforms.Normalize(std=(0.5, 0.5, 0.5),mean=(0.5, 0.5, 0.5))])
    cifar_train = datasets.CIFAR100(
        root="../data",train=True,transform=trans,download=False)
    cifar_test = datasets.CIFAR100(
        root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(cifar_train,batch_size,shuffle=True,
                            num_workers=4),
            data.DataLoader(cifar_test,batch_size,shuffle=False,
                            num_workers=4))

# 用于对多个变量进行累加
class Accumulator:
    # 创建n个空间
    def __init__(self, n):
        self.data = [0, 0] * n

    # 把原来类中对应位置的data和新传入的args做 a + float(b)加法操作然后重新赋给该位置的data
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    # 重新设置空间大小并初始化
    def reset(self):
        self.data = [0, 0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 分类精度
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每行按列找到最大值
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch(net,trainloader,loss,updater,device):
    if isinstance(net,nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for i, data in enumerate(trainloader, 0):
        X, y = data[0].to(device), data[1].to(device)
        updater.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# 使用数据迭代器data_iter可访问的数据集，评估net的精度
def evaluate_accuracy(net, data_iter):
    # 判断net是否属于
    if isinstance(net, nn.Module):
        # 模型进入评估模式
        net.eval()
    # 正确预测数、预测总数
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, epoch_num, loss, updater, trainloader, testloader,device):
    for epoch in range(epoch_num):
        train_metrics = train_epoch(net,trainloader, loss,updater,device)
        test_acc = evaluate_accuracy(net,testloader)
        print(train_metrics,test_acc)

    # 输出最终损失函数和训练精度
    print(test_acc*100,"%%")

def show(train_loss_history):
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()