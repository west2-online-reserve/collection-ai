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
            # datasets\cifar-10-batches-py\data_batch_1
            f'E:/projects/datasets/cifar-10-batches-py/data_batch_{i+1}')
        train_data.append(data[b'data'])
        train_label.append(data[b'labels'])
    test = unpickle('E:/projects/datasets/cifar-10-batches-py/test_batch')
    test_data = test[b'data']
    testlabel = test[b'labels']
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    train_data = train_data.reshape(
        (50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype("float32")
    train_data = train_data.reshape(-1, 32*32*3)
    test_data = test_data.reshape((10000, 3, 32, 32)).transpose(
        0, 2, 3, 1).astype("float32")
    test_data = test_data.reshape(-1, 32*32*3)
    return train_data, train_label, test_data, testlabel


def preprocess_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    train_data -= mean
    test_data -= mean
    return train_data, test_data


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, dout, cache):
        dx, x = dout, cache
        dx[x <= 0] = 0
        return dx

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        Hidden_layer = self.relu(np.dot(X, W1) + b1)
        scores = np.dot(Hidden_layer, W2) + b2

        if y is None:
            return scores

        loss = np.sum(-np.log(np.exp(scores[range(len(scores)), y]) /
                              np.sum(np.exp(scores), axis=1)))
        loss /= len(X)
        loss += 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))

        grads = {}

        dscores = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)
        dscores[range(len(scores)), y] -= 1
        dscores /= len(X)
        grads['W2'] = np.dot(Hidden_layer.T, dscores) + reg*W2
        grads['b2'] = np.sum(dscores, axis=0)
        dhidden = np.dot(dscores, W2.T)
        dhidden_relu = self.relu_backward(dhidden, Hidden_layer)
        grads['W1'] = np.dot(X.T, dhidden_relu) + reg*W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def train(self, X, y,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        for it in range(num_iters):
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            # if verbose and it % 100 == 0:
            #     print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            if it % iterations_per_epoch == 0 and it != 0:
                learning_rate *= learning_rate_decay

    def predict(self, X):
        Hidden_layer = self.relu(
            np.dot(X, self.params['W1']) + self.params['b1'])
        scores = np.dot(Hidden_layer, self.params['W2']) + self.params['b2']
        return np.argmax(scores, axis=1)


def main():
    train_data, train_label, test_data, testlabel = fetch_data()
    train_data, test_data = preprocess_data(train_data, test_data)
    net = NeuralNetwork(32*32*3, 100, 10)
    net.train(train_data, train_label, num_iters=10000,
              batch_size=200, verbose=True)
    print(f'train accuracy: {np.mean(net.predict(train_data) == train_label)}')
    return


if __name__ == '__main__':
    main()
