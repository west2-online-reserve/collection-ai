import torch
import torchvision
from matplotlib import pyplot
from data import my_dataset
from model import ResNet
from loss import custom_loss
# from utils import process_data

# process_data()
turns, batch_size, learning_rate = 25, 64, 0.002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# train_set = my_dataset('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\JPEGImages', train=True, transform=transform)
# predict_set = my_dataset('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\JPEGImages', train=False, transform=transform)
train_set = my_dataset('./VOCdevkit/VOC2012/JPEGImages', train=True, transform=transform)
predict_set = my_dataset('./VOCdevkit/VOC2012/JPEGImages', train=False, transform=transform)
# train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# predict_set_loader = torch.utils.data.DataLoader(predict_set, batch_size=batch_size, shuffle=False)
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
predict_set_loader = torch.utils.data.DataLoader(predict_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
model = ResNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.0005)
trainAccuracy = [0, ] * turns
trainIoU = [0, ] * turns
trainLoss = [0, ] * turns
testAccuracy = [0, ] * turns
testIoU = [0, ] * turns
testLoss = [0, ] * turns
torch.backends.cudnn.enabled=False

for i in range(turns):
    if i < 5:
        learning_rate = 0.002*(i+1)
        for p in optimizer.param_groups:
            p['lr']=learning_rate
    # if i == 40:
    #     learning_rate = 0.0001
    #     for p in optimizer.param_groups:
    #         p['lr']=learning_rate
    model.train()
    sum_loss, sum_accurate, sum_iou = 0.0, 0, 0.0
    for data, target in train_set_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_block = custom_loss(5, 0.5)
        batch_loss, iou, accurate_num = loss_block(output, target)
        sum_loss += batch_loss.item()
        sum_accurate += accurate_num
        sum_iou += iou
        batch_loss.backward()
        optimizer.step()
        # print("当前批次train执行成功")
    trainLoss[i] = sum_loss / train_set.get_sum_boxes()
    trainAccuracy[i] = sum_accurate / train_set.get_sum_boxes()
    trainIoU[i] = sum_iou / train_set.get_sum_boxes()
    print("Train %d: Average Loss: %.6f, Average IoU: %.2f %%, Accuracy: %.2f %%" % (
        i + 1, trainLoss[i], 100 * trainIoU[i], 100 * trainAccuracy[i]))
    model.eval()
    sum_loss, sum_accurate, sum_iou = 0.0, 0, 0.0
    with torch.no_grad():
        for data, target in predict_set_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_block = custom_loss(5, 0.5)
            batch_loss, iou, accurate_num = loss_block(output, target)
            sum_loss += batch_loss.item()
            sum_accurate += accurate_num
            sum_iou += iou
    testLoss[i] = sum_loss / predict_set.get_sum_boxes()
    testAccuracy[i] = sum_accurate / predict_set.get_sum_boxes()
    testIoU[i] = sum_iou / predict_set.get_sum_boxes()
    print("Test %d: Average Loss: %.6f, Average IoU: %.2f %%, Accuracy: %.2f %%" % (
        i + 1, testLoss[i], 100 * testIoU[i], 100 * testAccuracy[i]))

torch.save(model.state_dict(),'model.pth')
x = range(1, turns + 1)
pyplot.figure(1)
pyplot.title("训练集/测试集损失变化",fontproperties="SimSun")
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
# pyplot.ylim([0,5])
pyplot.plot(x, trainLoss, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testLoss, label='val', color='blue', linestyle='-.')
pyplot.legend()
pyplot.savefig('loss.jpg')

pyplot.figure(2)
pyplot.title("训练集/测试集正确率变化",fontproperties="SimSun")
pyplot.xlabel("epoch")
pyplot.ylabel("accuracy")
pyplot.ylim([0, 1])
pyplot.plot(x, trainAccuracy, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testAccuracy, label='val', color='blue', linestyle='-.')
pyplot.legend()
pyplot.savefig('accuracy.jpg')

pyplot.figure(3)
pyplot.title("训练集/测试集交并比变化",fontproperties="SimSun")
pyplot.xlabel("epoch")
pyplot.ylabel("IoU")
pyplot.ylim([0, 1])
pyplot.plot(x, trainIoU, label='train', color='purple', linestyle='-.')
pyplot.plot(x, testIoU, label='val', color='blue', linestyle='-.')
pyplot.legend()
pyplot.savefig('iou.jpg')

