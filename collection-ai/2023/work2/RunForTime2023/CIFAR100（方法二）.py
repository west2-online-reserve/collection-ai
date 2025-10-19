import torch
import torchvision
import pickle
from matplotlib import pyplot

class CIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(3, 20, 6), torch.nn.BatchNorm2d(20), torch.nn.ReLU())
        self.pool = torch.nn.Sequential(torch.nn.MaxPool2d(3, 3), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=1620, hidden_size=1600, batch_first=True)
        self.linear1 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1600, 1000), torch.nn.BatchNorm1d(1000), torch.nn.ReLU())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(1000, 1000), torch.nn.BatchNorm1d(1000), torch.nn.ReLU())
        self.linear3 = torch.nn.Linear(1000, 100)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x, y = self.gru(x.reshape(batchSize, 1, -1))  # 强行展平图片
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

batchSize = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learningRate = 0.01
turns = 10
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(0.5),
        # torchvision.transforms.Resize(75),
        # torchvision.transforms.Grayscale(1), # 改为单通道灰度图
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
trainingSet = torchvision.datasets.CIFAR100('CIFAR100', train=True, download=True, transform=transform)
predictSet = torchvision.datasets.CIFAR100('CIFAR100', train=False, download=True, transform=transform)
trainingSetLoader = torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True)
predictSetLoader = torch.utils.data.DataLoader(predictSet, batch_size=batchSize, shuffle=True)
model = CIFAR().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

with open("./CIFAR100/cifar-100-python/train", 'rb') as file:
    dict = pickle.load(file, encoding='latin1')
fineLabels = dict['fine_labels']
coarseLabels = dict['coarse_labels']
dict = {}
dict.update([(fine, coarse) for fine, coarse in zip(fineLabels, coarseLabels)])
trainLoss = [0, ] * turns
testLoss = [0, ] * turns
trainAccuracy = [0, ] * turns
testAccuracy = [0, ] * turns
temp1 = [0, ] * turns
temp2 = [0, ] * turns

for i in range(turns):
    model.train()
    accurateNum1 = accurateNum2 = loss2 = 0.0
    for data, label in trainingSetLoader:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(output, label)
        loss2 += loss
        result = output.max(1, keepdim=True)[1]
        for x, y in zip(result[:, 0], label):
            if dict[int(x)] == dict[int(y)]:
                accurateNum1 += 1
        temp = result[:, 0] == label
        accurateNum2 += int(temp.sum())
        loss.backward()
        optimizer.step()
    trainLoss[i] = loss2 / len(trainingSetLoader.dataset)
    trainAccuracy[i] = accurateNum1 / len(trainingSetLoader.dataset)
    temp1[i] = accurateNum2 / len(trainingSetLoader.dataset)

    model.eval()
    accurateNum1 = accurateNum2 = loss2 = 0.0
    with torch.no_grad():
        for data, label in predictSetLoader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()
            loss2 += loss(output, label)
            result = output.max(1, keepdim=True)[1]
            for x, y in zip(result[:, 0], label):
                if dict[int(x)] == dict[int(y)]:
                    accurateNum1 += 1
            temp = result[:, 0] == label
            accurateNum2 += int(temp.sum())
    testLoss[i] = loss2 / len(predictSetLoader.dataset)
    testAccuracy[i] = accurateNum1 / len(predictSetLoader.dataset)
    temp2[i] = accurateNum2 / len(predictSetLoader.dataset)

device = torch.device("cpu")
print("Results:\nTest Average Loss: %.6f\nTest Accuracy(superclass): %.2f %%\nTest Accuracy(subclass): %.2f %%" % (
testLoss[turns - 1], testAccuracy[turns - 1] * 100, temp2[turns - 1] * 100))
epoch = range(1, turns + 1)
axis1 = pyplot.subplot()
axis2 = axis1.twinx()
axis1.plot(epoch, torch.tensor(trainLoss).to(device), label='Train Loss', color='red', linestyle='-.')
axis1.plot(epoch, torch.tensor(testLoss).to(device), label='Test Loss', color='green', linestyle='-.')
axis2.plot(epoch, torch.tensor(trainAccuracy).to(device), label='Train Accuracy(superclass)', linestyle='-.')
axis2.plot(epoch, torch.tensor(testAccuracy).to(device), label='Test Accuracy(superclass)', linestyle='-.')
axis1.set_xlabel("epoch")
axis1.set_ylabel("loss")
axis2.set_ylabel("accuracy")
pyplot.ylim(0, 1)
pyplot.title("Results")
handle1, label1 = axis1.get_legend_handles_labels()
handle2, label2 = axis2.get_legend_handles_labels()
axis1.legend(handle1 + handle2, label1 + label2)  # 合并图例
pyplot.show()