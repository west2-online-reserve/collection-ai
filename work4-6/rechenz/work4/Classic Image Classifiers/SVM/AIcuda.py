import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from urllib.request import urlretrieve
import tarfile


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        super(TwoLayerNet, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 初始化权重
        self.fc1.weight.data = std * torch.randn(hidden_size, input_size)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data = std * torch.randn(output_size, hidden_size)
        self.fc2.bias.data.zero_()

        # 移动到GPU
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        hidden = F.relu(self.fc1(x))
        scores = self.fc2(hidden)
        return scores

    def svm_loss(self, scores, y, reg=0.0):
        # 计算SVM损失
        num_train = scores.size(0)
        scores_correct = scores[torch.arange(num_train), y]
        margins = torch.clamp(scores - scores_correct.view(-1, 1) + 1.0, min=0)
        margins[torch.arange(num_train), y] = 0
        loss = torch.sum(margins) / num_train
        loss += 0.5 * reg * (torch.sum(self.fc1.weight ** 2) +
                             torch.sum(self.fc2.weight ** 2))
        return loss

    def train_step(self, X, y, learning_rate=1e-3, reg=0.0):
        # 前向传播
        scores = self.forward(X)
        loss = self.svm_loss(scores, y, reg)

        # 反向传播
        self.zero_grad()
        loss.backward()

        # 更新参数
        with torch.no_grad():
            self.fc1.weight -= learning_rate * self.fc1.weight.grad
            self.fc1.bias -= learning_rate * self.fc1.bias.grad
            self.fc2.weight -= learning_rate * self.fc2.weight.grad
            self.fc2.bias -= learning_rate * self.fc2.bias.grad

        return loss.item()

    def predict(self, X):
        with torch.no_grad():
            scores = self.forward(X)
            return torch.argmax(scores, dim=1).cpu().numpy()


class CIFAR10Loader:
    def __init__(self):
        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.filename = 'cifar-10-python.tar.gz'
        self.data_dir = './cifar-10-batches-py'

    def download_and_extract(self):
        """下载并解压CIFAR-10数据集"""
        if not os.path.exists(self.filename):
            print('下载CIFAR-10数据集...')
            urlretrieve(self.url, self.filename)

        if not os.path.exists(self.data_dir):
            print('解压文件...')
            with tarfile.open(self.filename, 'r:gz') as tar:
                tar.extractall()

    def load_cifar10(self):
        """加载CIFAR-10数据"""
        self.download_and_extract()

        # 加载训练数据
        X_train = []
        y_train = []

        for i in range(1, 6):
            filename = os.path.join(self.data_dir, f'data_batch_{i}')
            with open(filename, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                X_train.append(data_dict[b'data'])
                y_train.append(data_dict[b'labels'])

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        # 加载测试数据
        filename = os.path.join(self.data_dir, 'test_batch')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            X_test = data_dict[b'data']
            y_test = np.array(data_dict[b'labels'])

        return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test):
    """数据预处理"""
    # 将数据转换为浮点型并归一化
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 计算均值并中心化
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # 转换为PyTorch张量
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    return X_train, X_test


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def fetch_data():
    train_data = []
    train_label = []
    test_data = []
    testlabel = []
    for i in range(5):
        data = unpickle(
            f'datasets/cifar-10-batches-py/data_batch_{i+1}')
        train_data.append(data[b'data'])
        train_label.append(data[b'labels'])
    test = unpickle('datasets/cifar-10-batches-py/test_batch')
    test_data = test[b'data']
    testlabel = test[b'labels']
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    train_data = train_data.reshape(
        (50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype("float")
    train_data = train_data.reshape(-1, 32*32*3)
    test_data = test_data.reshape((10000, 3, 32, 32)).transpose(
        0, 2, 3, 1).astype("float")
    test_data = test_data.reshape(-1, 32*32*3)
    return train_data, train_label, test_data, testlabel


def train(net, X_train, y_train, X_val, y_val,
          learning_rate=1e-3, learning_rate_decay=0.95,
          reg=5e-6, num_iters=100,
          batch_size=200, verbose=False):

    # 转换标签为PyTorch张量
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()

    num_train = X_train.size(0)
    iterations_per_epoch = max(num_train // batch_size, 1)

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        # 随机选择小批量
        batch_indices = torch.randperm(num_train)[:batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # 计算损失和更新参数
        loss = net.train_step(X_batch, y_batch, learning_rate, reg)
        loss_history.append(loss)

        if verbose and it % 100 == 0:
            print(f'迭代 {it}/{num_iters}: 损失 {loss}')

        # 每个epoch检查准确率并衰减学习率
        if it % iterations_per_epoch == 0:
            train_acc = (net.predict(X_train) == y_train.numpy()).mean()
            val_acc = (net.predict(X_val) == y_val.numpy()).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            learning_rate *= learning_rate_decay

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }


def main():
    # 加载数据
    print("加载CIFAR-10数据集...")
    X_train, y_train, X_test, y_test = fetch_data()

    # 数据预处理
    print("数据预处理...")
    X_train, X_test = preprocess_data(X_train, X_test)

    # 分割训练集和验证集
    num_training = 49000
    num_validation = 1000

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    # 初始化神经网络
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10

    print("初始化神经网络...")
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # 训练网络
    print("开始训练...")
    stats = train(net, X_train, y_train, X_val, y_val,
                  num_iters=10000, batch_size=200,
                  learning_rate=1e-3, learning_rate_decay=0.95,
                  reg=0.001, verbose=True)

    # 在测试集上评估
    test_acc = (net.predict(X_test) == y_test).mean()
    print(f'测试集准确率: {test_acc}')

    # 在训练集和验证集上评估
    val_acc = (net.predict(X_val) == y_val).mean()
    train_acc = (net.predict(X_train) == y_train).mean()
    print(f'训练集准确率: {train_acc}')
    print(f'验证集准确率: {val_acc}')


if __name__ == '__main__':
    main()
