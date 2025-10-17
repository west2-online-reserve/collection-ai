import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms


def load_data_mnist(batch_size, resize=None):
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0～1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # Compose将多个图像变换操作组成一个变换操作序列
    trans = transforms.Compose(trans)
    mnist_train = datasets.MNIST(
        root="../data", train=True, transform=trans,download=False)
    mnist_test = datasets.MNIST(
        root="../data", train=False, transform=trans,download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4))
# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        # w是以均值为0，标准差为0.01的正态分布
        nn.init.normal_(m.weight,std=0.01)

# 分类精度
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每行按列找到最大值
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 用于对多个变量进行累加
class Accumulator:
    # 创建n个空间
    def __init__(self, n):
        self.data = [0, 0] * n
    # 把原来类中对应位置的data和新传入的args做 a + float(b)加法操作然后重新赋给该位置的data
    def add(self, *args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    # 重新设置空间大小并初始化
    def reset(self):
        self.data = [0, 0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# 使用数据迭代器data_iter可访问的数据集，评估net的精度
def evaluate_accuracy(net, data_iter):
    # 判断net是否属于
    if isinstance(net,nn.Module):
        # 模型进入评估模式
        net.eval()
    # 正确预测数、预测总数
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

# 训练模型迭代一个周期
def train_epoch(net,train_iter,loss,updater):
    if isinstance(net,nn.Module):
        # 模型进入训练模式
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]


def train(net,train_iter,test_iter, loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics =train_epoch(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        print(train_metrics)
    train_loss,train_acc = train_metrics
    # 输出最终损失函数和训练精度
    # print(train_loss)
    # print(train_acc*100,"%%")
    # assert的作用是判断训练损失(train_loss)是否小于0.5，
    # 如果不满足条件就会抛出异常并打印出train_loss的值，以此类推
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

def get_mnist_labels(y):
    labels = []
    for i in range(0,10):
        labels+=str(i)
    return [labels[int(i)] for i in y]

# 对图像进行分类预测
def predict(net,test_iter,n=128):
    for X,y in test_iter:
        break
    trues = get_mnist_labels(y)
    preds = get_mnist_labels(net(X).argmax(axis=1))
    titles = [true+':'+pred for true ,pred in zip(trues,preds)]
    r = random.randint(0,len(X)-n)
    show_images(X[r:r+n].reshape(n,28,28),16,8,titles=titles[r:r+n])

def main():
    # 在linear线性层前定义展平层，将数组转换为向量
    net = nn.Sequential(nn.Flatten(), nn.Linear(784,10))
    net.apply(init_weights)# 每一层的权值都进行初始化
    # 损失函数使用交叉熵
    loss = nn.CrossEntropyLoss(reduction='none')
    # 优化器使用小批量随机梯度下降
    updater = torch.optim.SGD(net.parameters(), lr=0.47)
    # 读取数据集
    batch_size = 256
    train_iter, test_iter = load_data_mnist(batch_size)
    # 训练十轮
    num_epochs = 10
    train(net,train_iter,test_iter,loss,num_epochs,updater)
    # 预测128个测试集图片并且在图片上方显示真实值和预测值
    predict(net, test_iter)

if __name__ =='__main__':
    main()
