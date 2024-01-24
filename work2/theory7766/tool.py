import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import numpy

categories = {
    0: ['castle', 'skyscraper', 'bridge', 'house', 'road'],
    1: ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    2: ['bear', 'lion', 'tiger', 'wolf', 'leopard'],
    3: ['beetle', 'bee', 'butterfly', 'caterpillar', 'cockroach'],
    4: ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    5: ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    6: ['cattle', 'chimpanzee', 'camel', 'elephant', 'kangaroo'],
    7: ['maple_tree', 'oak_tree', 'pine_tree', 'palm_tree', 'willow_tree'],
    8: ['television', 'clock', 'keyboard', 'telephone', 'lamp'],
    9: ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    10: ['worm', 'spider', 'snail', 'lobster', 'crab'],
    11: ['plate', 'cup', 'bottle', 'bowl', 'can'],
    12: ['woman', 'baby', 'boy', 'girl', 'man'],
    13: ['skunk', 'fox', 'raccoon', 'possum', 'porcupine'],
    14: ['bus', 'pickup_truck', 'train', 'motorcycle', 'bicycle'],
    15: ['aquarium_fish', 'ray', 'flatfish', 'shark', 'trout'],
    16: ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    17: ['tank', 'tractor', 'streetcar', 'rocket', 'lawn_mower'],
    18: ['couch', 'chair', 'bed', 'wardrobe', 'table'],
    19: ['lizard','snake', 'turtle', 'dinosaur', 'crocodile']
}

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
# ToTensor()将数值归一化到[0,1]
# Normalize()使用均值和标准差对张量图片进行归一化,将数据分布到(-1,1)，均值为0，方差为1
# 加入随机左右翻转数据增强,用imagenet的方差做归一化
trans_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
trans_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
cifar_train = datasets.CIFAR100(
    root="../data", train=True, transform=trans_train, download=False)
cifar_test = datasets.CIFAR100(
    root="../data", train=False, transform=trans_test, download=False)

# 加载CIFAR-100数据集
def load_data_cifar(batch_size):
    # 进行大类转换
    trans_cifar_test()
    trans_cifar_train()
    return (data.DataLoader(cifar_train, batch_size, shuffle=True),
            data.DataLoader(cifar_test, batch_size, shuffle=False))

def trans_cifar_train():
    # 对50000张训练图进行预处理
    for i in range(1, 50000):
        name = cifar_train.classes[cifar_train.targets[i]]
        if name in categories[0]:
            cifar_train.targets[i] = 0
        elif name in categories[1]:
            cifar_train.targets[i] = 1
        elif name in categories[2]:
            cifar_train.targets[i] = 2
        elif name in categories[3]:
            cifar_train.targets[i] = 3
        elif name in categories[4]:
            cifar_train.targets[i] = 4
        elif name in categories[5]:
            cifar_train.targets[i] = 5
        elif name in categories[6]:
            cifar_train.targets[i] = 6
        elif name in categories[7]:
            cifar_train.targets[i] = 7
        elif name in categories[8]:
            cifar_train.targets[i] = 8
        elif name in categories[9]:
            cifar_train.targets[i] = 9
        elif name in categories[10]:
            cifar_train.targets[i] = 10
        elif name in categories[11]:
            cifar_train.targets[i] = 11
        elif name in categories[12]:
            cifar_train.targets[i] = 12
        elif name in categories[13]:
            cifar_train.targets[i] = 13
        elif name in categories[14]:
            cifar_train.targets[i] = 14
        elif name in categories[15]:
            cifar_train.targets[i] = 15
        elif name in categories[16]:
            cifar_train.targets[i] = 16
        elif name in categories[17]:
            cifar_train.targets[i] = 17
        elif name in categories[18]:
            cifar_train.targets[i] = 18
        elif name in categories[19]:
            cifar_train.targets[i] = 19

def trans_cifar_test():
    for i in range(1,10000):
        name = cifar_test.classes[cifar_test.targets[i]]
        if name in categories[0]:
            cifar_test.targets[i] = 0
        elif name in categories[1]:
            cifar_test.targets[i] = 1
        elif name in categories[2]:
            cifar_test.targets[i] = 2
        elif name in categories[3]:
            cifar_test.targets[i] = 3
        elif name in categories[4]:
            cifar_test.targets[i] = 4
        elif name in categories[5]:
            cifar_test.targets[i] = 5
        elif name in categories[6]:
            cifar_test.targets[i] = 6
        elif name in categories[7]:
            cifar_test.targets[i] = 7
        elif name in categories[8]:
            cifar_test.targets[i] = 8
        elif name in categories[9]:
            cifar_test.targets[i] = 9
        elif name in categories[10]:
            cifar_test.targets[i] = 10
        elif name in categories[11]:
            cifar_test.targets[i] = 11
        elif name in categories[12]:
            cifar_test.targets[i] = 12
        elif name in categories[13]:
            cifar_test.targets[i] = 13
        elif name in categories[14]:
            cifar_test.targets[i] = 14
        elif name in categories[15]:
            cifar_test.targets[i] = 15
        elif name in categories[16]:
            cifar_test.targets[i] = 16
        elif name in categories[17]:
            cifar_test.targets[i] = 17
        elif name in categories[18]:
            cifar_test.targets[i] = 18
        elif name in categories[19]:
            cifar_test.targets[i] = 19

def train(net, epoch_num, loss, updater, trainloader, testloader, device):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epoch_num):
        print("-----第{}轮训练开始------".format(epoch + 1))
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0.0, 0.0, 0.0, 0.0
        # 开始训练
        if isinstance(net, nn.Module):
            net.train()
        for i, data in enumerate(trainloader):
            X, label = data[0].to(device), data[1].to(device)
            updater.zero_grad()
            y_hat = net(X)
            print(epoch,y_hat,label)
            print(y_hat.shape,label.shape)
            mask = (label >= 20)
            label[mask] = 0
            l1 = loss(y_hat, label)
            l1.mean().backward()
            updater.step()
            # 计算每轮训练集的loss
            train_loss += l1.item()
            # 计算训练集精度
            _, predicted = torch.max(y_hat.data, 1)
            train_cor += (predicted == label).sum().item()
            train_sum += label.size(0)

        # 进入测试模式
        if isinstance(net,nn.Module):
            net.eval()
        for j, data in enumerate(testloader):
            X, label = data[0].to(device), data[1].to(device)
            y_hat = net(X)
            l2 = loss(y_hat, label)
            test_loss += l2.item()
            _, predicted = torch.max(y_hat.data, 1)
            test_cor += (predicted == label).sum().item()
            test_sum += label.size(0)

        train_loss_list.append(train_loss/i)
        train_acc_list.append(train_cor/train_sum * 100)
        test_loss_list.append(test_loss/j)
        test_acc_list.append(test_cor/test_sum * 100)
        print("Train loss:{}   Train accuracy:{}%  Test loss:{}  Test accuracy:{}%".format(
            train_loss/i, train_cor/train_sum * 100, test_loss/j, test_cor/test_sum * 100))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def show_acc(train_acc_list, test_acc_list):
    # 创建准确率画布
    plt.figure()
    plt.plot(range(len(train_acc_list)), train_acc_list, 'red')
    plt.plot(range(len(test_acc_list)), test_acc_list, 'green')
    # 添加图例
    plt.legend(['train accuracy', 'test accuracy'], fontsize=14, loc='best')
    # 设置横纵坐标
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accuracy rate(%)', fontsize=14)
    # 生成网格线
    plt.grid()
    plt.savefig('CIFAR100_figAccuracy_02_1')
    plt.show()

def show_loss(train_loss_list, test_loss_list):
    plt.figure()
    plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
    plt.legend(['train loss', 'test loss'], fontsize=14, loc='best')
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss value', fontsize=14)
    plt.grid()
    plt.savefig('CIFAR100_figLOSS_02_1')
    plt.show()
