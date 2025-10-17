import torch
import torchvision

# 下载mnist数据集
pipeline = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),  # 将图像转为tensor
     torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 正则化/归一化：防止数据过拟合
     ])  # 原始图像处理方法

# 定义超参数
batchSize = 16  # 单批次处理的图片数量
device = torch.device("cpu")  # 使用CPU训练
turns = 10  # 数据训练轮数

# 下载数据集
trainingSet = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=pipeline)
predictSet = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=pipeline)
trainingSetLoader = torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True)  # shuffle=true:打乱顺序
predictSetLoader = torch.utils.data.DataLoader(predictSet, batch_size=batchSize, shuffle=True)


# 构建网络模型
class digit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)  # 灰度图片的通道，输出通道，卷积核大小
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.fullyConnectedLayer1 = torch.nn.Linear(20 * 10 * 10, 500)  # 输入通道，输出通道
        self.fullyConnectedLayer2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入：batch_size*1(通道)*28*28（28*28为像素），输出：batch_size*10*(28-5+1)*(28-5+1)
        x = torch.nn.functional.relu(x)  # 激活函数 激活层
        x = torch.nn.functional.max_pool2d(x, 2, 2)  # 池化层：压缩图片（舍弃图片的部分信息），两个维度的步长均为2（即隔行/列取元素）。输入：batch_size*10*24*24，输出：batch_size*10*12*12

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)

        x = x.view(input_size, -1)  # 将矩阵按行展成向量（如4*4矩阵展为长为16的数组）；-1:自动计算维度。 20*10*10=2000
        x = self.fullyConnectedLayer1(x)  # 输入：batch_size*2000，输出：batch_size*500
        x = torch.nn.functional.relu(x)
        x = self.fullyConnectedLayer2(x)  # 输入：batch_size*500，输出：batch_size*10
        output = torch.nn.functional.log_softmax(x, dim=1)  # 计算分类后每个数字的概率
        return output


# 定义优化器
model = digit().to(device)
optimizer = torch.optim.Adam(model.parameters())


# 定义训练方法
def training(model, device, trainingLoader, optimizer, turns):  # 模型，GPU/CPU，训练数据，优化器，当前训练轮次
    model.train()
    for i, (data, label) in enumerate(trainingLoader):
        # 数据部署到CPU上
        data, label = data.to(device), label.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 预测
        output = model(data)
        # 计算损失
        loss = torch.nn.functional.cross_entropy(output, label)  # cross_entropy()：计算定类数据的交叉熵损失函数
        result = output.max(1, keepdim=True)  # 1表示维度。也可写为：result=output.argmax(dim=1)。该行可选
        # 反向传播
        loss.backward()
        optimizer.step()
        # 打印训练过程（可选）
        # if i % 1000 == 0:
        #     print("Epoch:", turns, "\tLoss: %.6f" % loss.item())


# 定义测试方法
def prediction(model, device, predictLoader):
    # 模型验证
    model.eval()
    # 正确率
    correctRate = 0.0
    # 测试损失
    loss = 0.0
    with torch.no_grad():  # 不计算梯度，不进行反向传播
        for data, label in predictLoader:
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            loss += torch.nn.functional.cross_entropy(output, label).item()
            # 找到概率值最大的下标
            result = output.max(1, keepdim=True)[1]  # 也可写作：result=torch.max(output,dim=1)。此处[0]为值，[1]为索引，我们需要的是索引
            # 累计正确的数量
            correctRate += result.eq(label.view_as(result)).sum().item()
        loss /= len(predictLoader.dataset)
        print("Test —— Average Loss:%.6f" % loss, ", Accuracy:", 100 * correctRate / len(predictLoader.dataset), "%")


for i in range(turns):
    training(model, device, trainingSetLoader, optimizer, i + 1)
    prediction(model, device, predictSetLoader)
