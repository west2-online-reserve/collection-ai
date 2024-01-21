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
    # print("Train %d —— Average Loss:%.6f, Accuracy(Superclass): %.2f %%, Accuracy(Subclass): %.2f %%" % (
    # i + 1, trainLoss[i], trainAccuracy[i] * 100, temp1[i] * 100))
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
    # print("Test", i + 1, "—— Average Loss:%.6f" % testLoss[i],
    #       ", Accuracy(Superclass):", testAccuracy[i] * 100,
    #       "%, Accuracy(Subclass):", temp2[i] * 100, "%")

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
# Train 1 —— Average Loss:0.225961, Accuracy(Superclass): 27.71 %, Accuracy(Subclass): 15.96 %
# Test 1 —— Average Loss:0.189819 , Accuracy(Superclass): 39.79 %, Accuracy(Subclass): 26.51 %
# Train 2 —— Average Loss:0.177910, Accuracy(Superclass): 42.85 %, Accuracy(Subclass): 29.33 %
# Test 2 —— Average Loss:0.162943 , Accuracy(Superclass): 47.05 %, Accuracy(Subclass): 34.18 %
# Train 3 —— Average Loss:0.155745, Accuracy(Superclass): 49.80 %, Accuracy(Subclass): 36.68 %
# Test 3 —— Average Loss:0.151629 , Accuracy(Superclass): 50.55 %, Accuracy(Subclass): 37.28 %
# Train 4 —— Average Loss:0.140854, Accuracy(Superclass): 54.47 %, Accuracy(Subclass): 41.66 %
# Test 4 —— Average Loss:0.136772 , Accuracy(Superclass): 56.13 %, Accuracy(Subclass): 43.419999999999995 %
# Train 5 —— Average Loss:0.128356, Accuracy(Superclass): 58.29 %, Accuracy(Subclass): 46.11 %
# Test 5 —— Average Loss:0.130159 , Accuracy(Superclass): 58.709999999999994 %, Accuracy(Subclass): 44.86 %
# Train 7 —— Average Loss:0.108635, Accuracy(Superclass): 64.82 %, Accuracy(Subclass): 53.47 %
# Test 7 —— Average Loss:0.122785 , Accuracy(Superclass): 60.38 %, Accuracy(Subclass): 47.21 %
# Train 8 —— Average Loss:0.099365, Accuracy(Superclass): 67.91 %, Accuracy(Subclass): 57.17 %
# Test 8 —— Average Loss:0.118262 , Accuracy(Superclass): 62.5 %, Accuracy(Subclass): 49.74 %
# Train 9 —— Average Loss:0.090520, Accuracy(Superclass): 70.90 %, Accuracy(Subclass): 60.88 %
# Test 9 —— Average Loss:0.115920 , Accuracy(Superclass): 63.38 %, Accuracy(Subclass): 50.82 %
# Train 10 —— Average Loss:0.082380, Accuracy(Superclass): 73.54 %, Accuracy(Subclass): 64.12 %
# Test 10 —— Average Loss:0.116297 , Accuracy(Superclass): 63.54 %, Accuracy(Subclass): 51.61 %
# Train 11 —— Average Loss:0.074904, Accuracy(Superclass): 75.80 %, Accuracy(Subclass): 67.17 %
# Test 11 —— Average Loss:0.117926 , Accuracy(Superclass): 63.44 %, Accuracy(Subclass): 50.92 %
# Train 12 —— Average Loss:0.067002, Accuracy(Superclass): 78.75 %, Accuracy(Subclass): 70.86 %
# Test 12 —— Average Loss:0.113055 , Accuracy(Superclass): 65.09 %, Accuracy(Subclass): 52.68000000000001 %
# Train 13 —— Average Loss:0.059570, Accuracy(Superclass): 81.14 %, Accuracy(Subclass): 74.05 %
# Test 13 —— Average Loss:0.117838 , Accuracy(Superclass): 63.27 %, Accuracy(Subclass): 51.51 %
# Train 14 —— Average Loss:0.053269, Accuracy(Superclass): 83.44 %, Accuracy(Subclass): 76.95 %
# Test 14 —— Average Loss:0.114524 , Accuracy(Superclass): 64.91 %, Accuracy(Subclass): 53.059999999999995 %
# Train 15 —— Average Loss:0.046616, Accuracy(Superclass): 85.64 %, Accuracy(Subclass): 79.80 %
# Test 15 —— Average Loss:0.114135 , Accuracy(Superclass): 65.44 %, Accuracy(Subclass): 53.480000000000004 %
# Train 16 —— Average Loss:0.040369, Accuracy(Superclass): 87.69 %, Accuracy(Subclass): 82.72 %
# Test 16 —— Average Loss:0.118604 , Accuracy(Superclass): 64.45 %, Accuracy(Subclass): 52.89 %
# Train 17 —— Average Loss:0.035306, Accuracy(Superclass): 89.38 %, Accuracy(Subclass): 84.97 %
# Test 17 —— Average Loss:0.119998 , Accuracy(Superclass): 65.14 %, Accuracy(Subclass): 53.32 %
# Train 18 —— Average Loss:0.030962, Accuracy(Superclass): 90.88 %, Accuracy(Subclass): 86.95 %
# Test 18 —— Average Loss:0.118357 , Accuracy(Superclass): 65.12 %, Accuracy(Subclass): 53.2 %
# Train 19 —— Average Loss:0.026911, Accuracy(Superclass): 92.30 %, Accuracy(Subclass): 88.90 %
# Test 19 —— Average Loss:0.121964 , Accuracy(Superclass): 64.75 %, Accuracy(Subclass): 52.6 %
# Train 20 —— Average Loss:0.022609, Accuracy(Superclass): 93.75 %, Accuracy(Subclass): 91.03 %
# Test 20 —— Average Loss:0.119170 , Accuracy(Superclass): 65.46 %, Accuracy(Subclass): 53.49 %
# Results:
# Test Average Loss: 0.119170
# Test Accuracy(superclass): 65.46 %
# Test Accuracy(subclass): 53.49 %