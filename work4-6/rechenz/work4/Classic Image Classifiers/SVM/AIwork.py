import numpy as np

np.random.seed(42)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        初始化两层神经网络
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def relu_backward(self, dout, cache):
        """ReLU反向传播"""
        x = cache
        dx = dout * (x > 0)
        return dx

    def loss(self, X, y=None, reg=0.0):
        """
        计算前向传播和损失
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 前向传播
        hidden_layer = self.relu(np.dot(X, W1) + b1)  # (N, H)
        scores = np.dot(hidden_layer, W2) + b2        # (N, C)

        if y is None:
            return scores

        # 计算SVM损失
        num_train = X.shape[0]
        scores_correct = scores[np.arange(num_train), y]  # 正确类别的得分
        margins = np.maximum(
            0, scores - scores_correct[:, np.newaxis] + 1.0)  # 合页损失
        margins[np.arange(num_train), y] = 0  # 正确类别不计损失

        loss = np.sum(margins) / num_train
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # L2正则化

        # 反向传播
        grads = {}

        # 计算梯度
        dscores = np.zeros_like(scores)
        dscores[margins > 0] = 1
        dscores[np.arange(num_train), y] -= np.sum(dscores, axis=1)
        dscores /= num_train

        # 第二层梯度
        grads['W2'] = np.dot(hidden_layer.T, dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        # 第一层梯度
        dhidden = np.dot(dscores, W2.T)
        dhidden_relu = self.relu_backward(dhidden, hidden_layer)
        grads['W1'] = np.dot(X.T, dhidden_relu) + reg * W1
        grads['b1'] = np.sum(dhidden_relu, axis=0)

        return loss, grads

    def train(self, X, y,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        使用随机梯度下降训练网络
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # 随机选择小批量
            batch_indices = np.random.choice(
                num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # 计算损失和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            # 更新参数
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print(f'迭代 {it}/{num_iters}: 损失 {loss}')

            # 每个epoch检查准确率并衰减学习率
            if it % iterations_per_epoch == 0:
                # 衰减学习率
                learning_rate *= learning_rate_decay

    def predict(self, X):
        """
        预测输入数据的标签
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer = self.relu(np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        y_pred = np.argmax(scores, axis=1)

        return y_pred


def preprocess_data(X_train, X_test):
    """数据预处理"""
    # 将数据转换为浮点型并归一化
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 计算均值并中心化
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # 将数据重塑为 (N, 3, 32, 32) 然后展平为 (N, 3072)
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0,
                                                       2, 3, 1).reshape(-1, 32*32*3)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0,
                                                     2, 3, 1).reshape(-1, 32*32*3)

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
            # datasets\cifar-10-batches-py\data_batch_1
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


def main():
    # 加载数据
    print("加载CIFAR-10数据集...")
    X_train, y_train, X_test, y_test = fetch_data()

    # 数据预处理
    print("数据预处理...")
    X_train, X_test = preprocess_data(X_train, X_test)

    # 初始化神经网络
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10

    print("初始化神经网络...")
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # 训练网络
    print("开始训练...")
    net.train(X_train, y_train,
              num_iters=3000, batch_size=200,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=0.001, verbose=True)

    # 在测试集上评估
    test_acc = (net.predict(X_test) == y_test).mean()
    print(f'测试集准确率: {test_acc}')


if __name__ == '__main__':
    main()
