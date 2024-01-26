import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import numpy

categories = {
    'castle': 0, 'skyscraper': 0, 'bridge': 0, 'house': 0, 'road': 0,
    'beaver': 1, 'dolphin': 1, 'otter': 1, 'seal': 1, 'whale': 1,
    'bear': 2, 'lion': 2, 'tiger': 2, 'wolf': 2, 'leopard': 2,
    'beetle': 3, 'bee': 3, 'butterfly': 3, 'caterpillar': 3, 'cockroach': 3,
    'apple': 4, 'mushroom': 4, 'orange': 4, 'pear': 4, 'sweet_pepper': 4,
    'cloud': 5, 'forest': 5, 'mountain': 5, 'plain': 5, 'sea': 5,
    'cattle': 6, 'chimpanzee': 6, 'camel': 6, 'elephant': 6, 'kangaroo': 6,
    'maple_tree': 7, 'oak_tree': 7, 'pine_tree': 7, 'palm_tree': 7, 'willow_tree': 7,
    'television': 8, 'clock': 8, 'keyboard': 8, 'telephone': 8, 'lamp': 8,
    'hamster': 9, 'mouse': 9, 'rabbit': 9, 'shrew': 9, 'squirrel': 9,
    'worm': 10, 'spider': 10, 'snail': 10, 'lobster': 10, 'crab': 10,
    'plate': 11, 'cup': 11, 'bottle': 11, 'bowl': 11, 'can': 11,
    'woman': 12, 'baby': 12, 'boy': 12, 'girl': 12, 'man': 12,
    'skunk': 13, 'fox': 13, 'raccoon': 13, 'possum': 13, 'porcupine': 13,
    'bus': 14, 'pickup_truck': 14, 'train': 14, 'motorcycle': 14, 'bicycle': 14,
    'aquarium_fish': 15, 'ray': 15, 'flatfish': 15, 'shark': 15, 'trout': 15,
    'orchid': 16, 'poppy': 16, 'rose': 16, 'sunflower': 16, 'tulip': 16,
    'tank': 17, 'tractor': 17, 'streetcar': 17, 'rocket': 17, 'lawn_mower': 17,
    'couch': 18, 'chair': 18, 'bed': 18, 'wardrobe': 18, 'table': 18,
    'lizard': 19, 'snake': 19, 'turtle': 19, 'dinosaur': 19, 'crocodile': 19
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
    root="../data", train=True, transform=trans_train, download=True)
cifar_test = datasets.CIFAR100(
    root="../data", train=False, transform=trans_test, download=True)


def trans_cifar_train():
    cifar_train.targets = \
        [categories[cifar_train.classes[cifar_train.targets[i]]] for i in range(len(cifar_train.targets))]


def trans_cifar_test():
    cifar_test.targets = \
        [categories[cifar_test.classes[cifar_test.targets[i]]] for i in range(len(cifar_test.targets))]


# 加载CIFAR-100数据集
def load_data_cifar(batch_size):
    # 进行大类转换
    trans_cifar_test()
    trans_cifar_train()
    return (data.DataLoader(cifar_train, batch_size, shuffle=True),
            data.DataLoader(cifar_test, batch_size, shuffle=False))


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
            # print(epoch,y_hat,label)
            # print(y_hat.shape,label.shape)
            # mask = (label >= 20)
            # label[mask] = 0
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
        if isinstance(net, nn.Module):
            net.eval()
        for j, data in enumerate(testloader):
            X, label = data[0].to(device), data[1].to(device)
            y_hat = net(X)
            l2 = loss(y_hat, label)
            test_loss += l2.item()
            _, predicted = torch.max(y_hat.data, 1)
            test_cor += (predicted == label).sum().item()
            test_sum += label.size(0)

        train_loss_list.append(train_loss / i)
        train_acc_list.append(train_cor / train_sum * 100)
        test_loss_list.append(test_loss / j)
        test_acc_list.append(test_cor / test_sum * 100)
        print("Train loss:{}   Train accuracy:{}%  Test loss:{}  Test accuracy:{}%".format(
            train_loss / i, train_cor / train_sum * 100, test_loss / j, test_cor / test_sum * 100))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def show(train_acc_list, test_acc_list, train_loss_list, test_loss_list):
    # 创建准确率画布
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1 = plt.plot(range(len(train_acc_list)), train_acc_list, 'red')
    line2 = plt.plot(range(len(test_acc_list)), test_acc_list, 'green')
    # 设置横纵坐标
    ax.set_xlabel('epochs', fontsize=14)
    ax.set_ylabel('accuracy rate(%)', fontsize=14)
    # 共用x轴
    ax2 = ax.twinx()
    line3 = ax2.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    line4 = ax2.plot(range(len(test_loss_list)), test_loss_list, 'yellow')
    ax2.set_ylabel('loss value', fontsize=14)
    # 合并图例
    # lines = line1+line2+line3+line4
    # labs = [l.get_label() for l in lines]
    # ax.legend(lines, labs, loc=0)
    ax.legend(['train accuracy', 'test accuracy'], loc='best')
    ax2.legend(['train loss', 'test loss'], loc='best')
    # 生成网格线
    plt.grid()
    plt.savefig('CIFAR100_fig_02_1')
    plt.show()
