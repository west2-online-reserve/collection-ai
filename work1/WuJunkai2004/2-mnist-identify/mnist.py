import torch
import numpy as np
import gzip
import os

# 加载MNIST数据集的函数
def load_mnist_data():
    # 定义文件路径
    data_dir = "./mnist"
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    # 加载训练图像
    with gzip.open(train_images_path, "rb") as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    # 加载训练标签
    with gzip.open(train_labels_path, "rb") as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 加载测试图像
    with gzip.open(test_images_path, "rb") as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    # 加载测试标签
    with gzip.open(test_labels_path, "rb") as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_images, train_labels, test_images, test_labels

# 加载MNIST数据集
train_images, train_labels, test_images, test_labels = load_mnist_data()

# 将数据转换为PyTorch张量并进行归一化
train_images = torch.from_numpy(train_images).unsqueeze(1).float() / 256.0
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images).unsqueeze(1).float() / 256.0
test_labels = torch.from_numpy(test_labels).long()

# 定义神经网络模型
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28 * 28, 256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = NeuralNetwork()

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 训练
num_epochs = 150
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_images)
    loss = loss_fn(outputs, train_labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 测试模型
with torch.no_grad():
    test_outputs = model(test_images)
    _, predicted_labels = torch.max(test_outputs, 1)

    # 计算准确率
    total = test_labels.size(0)
    correct = (predicted_labels == test_labels).sum().item()
    accuracy = correct / total

# 打印结果
print("正确数量:", correct)
print("错误数量:", total - correct)
print("准确率:", accuracy)

'''

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
accuracy = correct / total

# 打印结果
print("正确数量:", correct)
print("错误数量:", total - correct)
print("准确率:", accuracy)'''