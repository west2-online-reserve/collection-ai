import torch
from torch.nn import functional as F
from torch import optim
import torchvision
from torch import nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# batch_size = 512
input_size = 28  # 图片尺寸
num_classes = 10  # 标签种类
num_epochs = 3  # 训练总周期
batch_size = 64  # 批次大小

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root="mnist_data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root="mnist_data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


# 构建神经网络模型
class Net(nn.Module):
    def __init__(self):
        # 1. xw+b，创建三个全连接层
        # super(Net, self).__init__()
        # self.fc1 = torch.nn.Linear(28 * 28, 256)
        # self.fc2 = torch.nn.Linear(256, 64)
        # self.fc3 = torch.nn.Linear(64, 10)
        # self.dropout = nn.Dropout(0.5)
        # def forward(self, x):
        #     # relu：激活函数
        #     x = F.relu(self.fc1(x))
        #     x = F.relu(self.fc2(x))
        #     x = self.fc3(x)
        #     return x

        # 2.CNN
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # 1,28,28
            nn.Conv2d(
                in_channels=1,  # 灰度图
                out_channels=16,  # 要得到几个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小和原来一样，stride=1,padding=(kernel-1)/2,默认向下取整
            ),  # 16,28,28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化，输出结果为16,14,14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                5,
                1,
                2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                32,
                5,
                1,
                2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32,7,7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # 64,7,7
        )
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# 优化器
def get_model():
    model = Net()
    return model, optim.Adam(model.parameters(), lr=0.001)


# 热值化处理
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


# 画图
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color="blue")
    plt.legend(["value"], loc="upper right")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.show()


def fit(epoch, model, opt, train_loader):
    train_loss = []
    for step in range(epoch):
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            y_onehot = one_hot(y)
            loss = F.mse_loss(out, y_onehot)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())

            if batch_idx % 100 == 0:
                print(step, batch_idx, loss.item())
    plot_curve(train_loss)


model, opt = get_model()
fit(3, model, opt, train_loader)

# 测试
def test_right(test_loader):
    total_correct = 0
    for x, y in test_loader:
        out = model(x)
        # out: [b, 10] => pred: [b]
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct

    total_num = len(test_loader.dataset)
    acc = total_correct / total_num
    print("test acc:", acc)


test_right(test_loader)
