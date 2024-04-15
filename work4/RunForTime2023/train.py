import torch
import torchvision
from matplotlib import pyplot
from data import myDataset
from model import ResNet
from loss import custom_loss
from utils import process

# process()
turns, batchSize, learningRate = 10, 16, 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# trainingSet = myDataset('C:\\Users\\Classic\\Desktop', train=True, transform=transform)
# predictSet = myDataset('C:\\Users\\Classic\\Desktop', train=False, transform=transform)
trainingSet = myDataset('./', train=True, transform=transform)
predictSet = myDataset('./', train=False, transform=transform)
# trainingSetLoader = torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True)
# predictSetLoader = torch.utils.data.DataLoader(predictSet, batch_size=batchSize, shuffle=False)
trainingSetLoader = torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True, num_workers=4,
                                                pin_memory=True)
predictSetLoader = torch.utils.data.DataLoader(predictSet, batch_size=batchSize, shuffle=False, num_workers=4,
                                               pin_memory=True)
model = ResNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
trainAccuracy = [0, ] * turns
trainIoU = [0, ] * turns
trainLoss = [0, ] * turns
testAccuracy = [0, ] * turns
testIoU = [0, ] * turns
testLoss = [0, ] * turns

for i in range(turns):
    model.train()
    accurateNum, loss2, sum_iou = 0, 0.0, 0.0
    for data, target in trainingSetLoader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss1 = custom_loss(7, 2, 5, 0.5, device)
        concrete_loss, iou, accurateNum = loss1(output, target)
        loss2 += concrete_loss
        sum_iou += iou
        concrete_loss.backward()
        optimizer.step()
        # print(loss2.shape,sum_iou.shape,accurateNum.shape)
        # print("当前批次train执行成功")
    trainLoss[i] = loss2 / len(trainingSet)
    trainAccuracy[i] = accurateNum / len(trainingSet)
    trainIoU[i] = sum_iou / len(trainingSet)
    print("Train %d: Average Loss: %.6f, Average IoU: %.2f %%, Accuracy: %.2f %%" % (
        i + 1, trainLoss[i], 100*trainIoU[i], 100*trainAccuracy[i]))
    model.eval()
    accurateNum, loss2, sum_iou = 0, 0.0, 0.0
    with torch.no_grad():
        for data, target in predictSetLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss1 = custom_loss(7, 2, 5, 0.5,device)
            concrete_loss, iou, accurateNum = loss1(output, target)
            loss2 += concrete_loss
            sum_iou += iou
    testLoss[i] = loss2 / len(predictSet)
    testAccuracy[i] = accurateNum / len(predictSet)
    testIoU[i] = sum_iou / len(predictSet)
    print("Test %d: Average Loss: %.6f, Average IoU: %.2f %%, Accuracy: %.2f %%" % (
        i + 1, testLoss[i], 100 * testIoU[i], 100*testAccuracy[i]))

x = range(1, turns + 1)
pyplot.figure(1)
pyplot.title("训练集/测试集损失变化")
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.plot(x, trainLoss, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testLoss, label='val', color='blue', linestyle='-.')

pyplot.figure(2)
pyplot.title("训练集/测试集正确率变化")
pyplot.xlabel("epoch")
pyplot.ylabel("accuracy")
pyplot.ylim([0, 1])
pyplot.plot(x, trainAccuracy, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testAccuracy, label='val', color='blue', linestyle='-.')

pyplot.figure(3)
pyplot.title("训练集/测试集交并比变化")
pyplot.xlabel("epoch")
pyplot.ylabel("IoU")
pyplot.ylim([0, 1])
pyplot.plot(x, trainIoU, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testIoU, label='val', color='blue', linestyle='-.')
pyplot.show()
