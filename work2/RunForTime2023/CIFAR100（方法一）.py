import torch
import torchvision
import pickle
from matplotlib import pyplot

class CIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 70, 5), torch.nn.BatchNorm2d(70), torch.nn.LeakyReLU())
        self.pool1 = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2), torch.nn.LeakyReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(70, 160, 5), torch.nn.BatchNorm2d(160), torch.nn.LeakyReLU())
        self.pool2 = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2), torch.nn.LeakyReLU())
        self.linear1 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4000, 3000), torch.nn.BatchNorm1d(3000),torch.nn.LeakyReLU())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(3000, 2400), torch.nn.BatchNorm1d(2400), torch.nn.Sigmoid())
        self.linear3 = torch.nn.Linear(2400, 100)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

batchSize = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learningRate = 0.01
turns = 20
transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.5, 0.5)])
trainingSet = torchvision.datasets.CIFAR100('CIFAR100', train=True, download=True, transform=transform)
predictSet = torchvision.datasets.CIFAR100('CIFAR100', train=False, download=True, transform=transform)
trainingSetLoader = torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True)
predictSetLoader = torch.utils.data.DataLoader(predictSet, batch_size=batchSize, shuffle=True)
model = CIFAR().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

with open("./CIFAR100/cifar-100-python/train", 'rb') as file:
    dict1 = pickle.load(file, encoding='latin1')
fineLabels = dict1['fine_labels'] # 提取小类标签
coarseLabels = dict1['coarse_labels']  # 提取大类标签
dict2 = {}
dict2.update([(fine, coarse) for fine, coarse in zip(fineLabels, coarseLabels)])
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
            if dict2[int(x)] == dict2[int(y)]:
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
                if dict2[int(x)] == dict2[int(y)]:
                    accurateNum1 += 1
            temp = result[:, 0] == label
            accurateNum2 += int(temp.sum())
    testLoss[i] = loss2 / len(predictSetLoader.dataset)
    testAccuracy[i] = accurateNum1 / len(predictSetLoader.dataset)
    temp2[i] = accurateNum2 / len(predictSetLoader.dataset)

device = torch.device("cpu")
print("Results:\nTest Average Loss: %.6f\nTest Accuracy(superclass): %.2f %%\nTest Accuracy(subclass): %.2f %%" % (testLoss[turns - 1], testAccuracy[turns - 1] * 100, temp2[turns - 1] * 100))
epoch = range(1, turns + 1)
axis1 = pyplot.subplot()
pyplot.xticks(range(0, turns+1, 2))
axis2 = axis1.twinx()
axis1.plot(epoch, torch.tensor(trainLoss).to(device), label='Train Loss', color='red', linestyle='-.')
axis1.plot(epoch, torch.tensor(testLoss).to(device), label='Test Loss', color='green', linestyle='-.')
axis2.plot(epoch, torch.tensor(trainAccuracy).to(device), label='Train Accuracy(superclass)', linestyle='-.')
axis2.plot(epoch, torch.tensor(testAccuracy).to(device), label='Test Accuracy(superclass)', linestyle='-.')
# axis2.plot(epoch, torch.tensor(temp1).to(device), label='Train Accuracy(subclass)', color='brown', linestyle='-.')
# axis2.plot(epoch, torch.tensor(temp2).to(device), label='Test Accuracy(subclass)', color='purple', linestyle='-.')
axis1.set_xlabel("epoch")
axis1.set_ylabel("loss")
axis2.set_ylabel("accuracy")
pyplot.ylim(0, 1)
pyplot.title("Results")
handle1, label1 = axis1.get_legend_handles_labels()
handle2, label2 = axis2.get_legend_handles_labels()
axis1.legend(handle1 + handle2, label1 + label2)  # 合并图例
pyplot.show()
