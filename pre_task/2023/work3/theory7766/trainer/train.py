from torch import nn
import torch

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

