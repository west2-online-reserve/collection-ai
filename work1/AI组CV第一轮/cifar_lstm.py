import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
import time
import warnings
from torch import nn
from torch.nn import functional as F
from model import Attention,drop_path,ResNet18LSTM
import os
device = "cuda:0"
if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image
from lstm1 import LSTM2D
data_dir = "D:/Aenvironment/CIFAR100"
train_dir = data_dir
valid_dir = data_dir + "/test"
class CifarDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image.open(self.img[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label

    def load_annotations(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos

train_transforms=torchvision.transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1
                ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B

                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # 均值，标准差
            ]
        )
test_transforms=torchvision.transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1
                ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B

                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # 均值，标准差
            ]
        )
train_dataset = CifarDataset(root_dir=train_dir, ann_file = 'D:\\Aenvironment\\CIFAR100\\train_list_coarse.txt', transform=train_transforms)
test_dataset = CifarDataset(root_dir=train_dir, ann_file = 'D:\\Aenvironment\\CIFAR100\\test_list_coarse.txt', transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


model_ft = ResNet18LSTM()
filename = "best22.pt"
checkpoint = torch.load(filename)
best_acc = checkpoint["best_acc"]
model_ft.load_state_dict(checkpoint["state_dict"])
print(model_ft)


# feature_extract = True
# for param in model_ft.parameters():
#         param.requires_grad = False
# model_ft.output_Layer=nn.Sequential(
#             nn.Linear(1000, 20),
#             # nn.Softmax(dim=-1)
#         )
# 保存文件的名字

# checkpoint = torch.load(filename)
#
# best_acc = checkpoint["best_acc"]
# model_ft.load_state_dict(checkpoint["state_dict"])
# GPU还是CPU计算
model_ft = model_ft.to(device)
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t", name)
# else:
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t", name)

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数


def train(epoches):
    best_acc = 0.0
    since = time.time()
    for epoch in range(epoches):
        print("Epoch {}/{}".format(epoch, epoches))
        print("-" * 10)
        running_loss = 0.0
        running_corrects = 0
        total = 0.0
        model_ft.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零
            optimizer_ft.zero_grad()
            # 只有训练的时候计算和更新梯度
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            loss.backward()
            optimizer_ft.step()
            # 计算损失
            running_loss += loss.item()  # 0表示batch那个维度
            running_corrects += predicted.eq(labels).sum().item()  # 预测结果最大的和真实值是否一致

        epoch_loss = running_loss / (batch_idx + 1)  # 算平均
        epoch_acc = running_corrects * 100 / total
        time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间
        print(
            "Time elapsed {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            state = {
                "state_dict": model_ft.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                "best_acc": best_acc,
                "optimizer": optimizer_ft.state_dict(),
            }
            torch.save(state, filename)

        print()


def test(epoches):
    since = time.time()
    train_acc_history = []
    train_losses = []

    for epoch in range(epoches):
        print("Epoch {}/{}".format(epoch, epoches))
        print("-" * 10)
        model_ft.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                # 计算损失
                total += labels.size(0)
                running_loss += loss.item()  # 0表示batch那个维度
                running_corrects += predicted.eq(labels).sum().item()  # 预测结果最大的和真实值是否一致
                epoch_loss = running_loss / (batch_idx + 1)  # 算平均
                epoch_acc = running_corrects * 100 / total
                time_elapsed = time.time() - since
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
            print(
                    "Time elapsed {:.0f}m {:.0f}s".format(
                        time_elapsed // 60, time_elapsed % 60
                    )
                )
            print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

        print()

    return train_acc_history,train_losses,epoches

# train(5)


#
# # 加载模型
# checkpoint = torch.load(filename)
#
# best_acc = checkpoint["best_acc"]
# model_ft.load_state_dict(checkpoint["state_dict"])
acc,loss,epochs=test(1)


print(acc)
print(loss)
fig, ax1 = plt.subplots()




# 在训练的时候没注意到，只能写个类似的双y轴充数了。。。

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='b')
ax1.plot( acc, color='b', marker='o', label='Accuracy')

# 创建第二个Y轴对象
ax2 = ax1.twinx()


ax2.set_ylabel('Loss', color='r')
ax2.plot( loss, color='r', marker='x', label='Loss')

# 设置图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')

# 添加标题
plt.title('Accuracy and Loss')

# 显示图形
plt.show()
