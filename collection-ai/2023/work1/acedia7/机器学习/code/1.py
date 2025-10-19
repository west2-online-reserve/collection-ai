import torch
import torch.nn as nn
import torch.optim as opt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchsummary import summary


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# 加载MNIST数据集
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

# 定义超参数
batch_size = 64
learning_rate = 1e-3
epochs = 10

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型并打印模型结构
model = Net()
summary(model, (1, 28, 28))

# 定义优化器、学习率衰减和损失函数
optimizer = opt.Adam(model.parameters(), lr=learning_rate)
scheduler = opt.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
criterion = nn.NLLLoss()

# 训练模型
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印准确率和学习率
    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Accuracy: {:.2f}%, Learning Rate: {:.6f}'.format(epoch+1, epochs, accuracy, optimizer.param_groups[0]['lr']))

    # 更新学习率
    scheduler.step()

    # 如果准确率达到90%，停止训练
    if accuracy >= 90:
        break



