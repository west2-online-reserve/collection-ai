import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 初始化权重为float32
        nn.init.normal_(self.fc1.weight, std=std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=std)
        nn.init.zeros_(self.fc2.bias)

        # 确保权重是float32类型
        self.fc1.weight.data = self.fc1.weight.data.float()
        self.fc1.bias.data = self.fc1.bias.data.float()
        self.fc2.weight.data = self.fc2.weight.data.float()
        self.fc2.bias.data = self.fc2.bias.data.float()

        # 移动到GPU
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device).float()  # 确保输入是float32
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


def preprocess_data(train_data, test_data):
    # 归一化
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    # 计算均值并中心化
    mean_image = np.mean(train_data, axis=0)
    train_data = train_data.astype(np.float32) - mean_image.astype(np.float32)
    test_data = test_data.astype(np.float32) - mean_image.astype(np.float32)

    # 转换为PyTorch张量
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    return train_data, test_data


def train(model, X, y, learning_rate=1e-3, learning_rate_decay=0.95,
          reg=1e-4, num_iters=100, batch_size=200, verbose=False):
    # 转换标签为PyTorch张量
    y = torch.from_numpy(y).long().to(model.device)
    X = X.float().to(model.device)  # 确保输入数据是float32

    num_train = X.size(0)
    iter_per_epoch = max(num_train // batch_size, 1)

    for it in range(num_iters):
        # 随机选择小批量
        idx = torch.randperm(num_train)[:batch_size]
        X_batch = X[idx]
        y_batch = y[idx]

        # 训练步骤
        loss = model.train_step(X_batch, y_batch, learning_rate, reg)

        if verbose and it % iter_per_epoch == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            if it > 0:
                learning_rate *= learning_rate_decay


def main():
    # 加载数据
    train_data, train_label, test_data, testlabel = fetch_data()
    train_data, test_data = preprocess_data(train_data, test_data)

    # 分割验证集
    num_val = 1000
    X_val = train_data[:num_val]
    y_val = train_label[:num_val]
    X_train = train_data[num_val:]
    y_train = train_label[num_val:]

    # 初始化模型
    model = NeuralNetwork(32 * 32 * 3, 100, 10)

    # 训练模型
    train(model, X_train, y_train, num_iters=20000, batch_size=256, verbose=True)

    # 评估模型
    test_pred = model.predict(test_data)
    print('Test accuracy: %f' % (np.mean(test_pred == testlabel)))

    val_pred = model.predict(X_val)
    print('Validation accuracy: %f' % (np.mean(val_pred == y_val)))

    train_pred = model.predict(X_train)
    print('Training accuracy: %f' % (np.mean(train_pred == y_train)))


if __name__ == '__main__':
    main()
