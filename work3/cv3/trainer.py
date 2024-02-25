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
from vit_model1 import vit_base_patch32_224
import cv2
from torch.nn import functional as F

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

# batch_size = 64
# data_dir = "/tmp/pycharm_project_482/CIFAR100/CIFAR100"
# train_dir = data_dir
# valid_dir = data_dir + "/test"
# class CifarDataset(Dataset):
#     def __init__(self, root_dir, ann_file, transform=None):
#         self.ann_file = ann_file
#         self.root_dir = root_dir
#         self.img_label = self.load_annotations()
#         self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
#         self.label = [label for label in list(self.img_label.values())]
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.img)
#
#     def __getitem__(self, idx):
#         image = Image.open(self.img[idx])
#         label = self.label[idx]
#         if self.transform:
#             image = self.transform(image)
#         label = torch.from_numpy(np.array(label))
#         return image, label
#
#     def load_annotations(self):
#         data_infos = {}
#         with open(self.ann_file) as f:
#             samples = [x.strip().split(' ') for x in f.readlines()]
#             for filename, gt_label in samples:
#                 data_infos[filename] = np.array(gt_label, dtype=np.int64)
#         return data_infos
#
# train_transforms=torchvision.transforms.Compose(
#             [
#                 transforms.Resize(224),
#                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
#                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
#                 transforms.ColorJitter(
#                     brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1
#                 ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
#                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#                 ),  # 均值，标准差
#             ]
#         )
# test_transforms=torchvision.transforms.Compose(
#             [
#                 transforms.Resize(224),
#                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
#                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
#                 transforms.ColorJitter(
#                     brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1
#                 ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
#                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#                 ),  # 均值，标准差
#             ]
#         )
# train_dataset = CifarDataset(root_dir=train_dir, ann_file = '/tmp/pycharm_project_482/CIFAR100/CIFAR100/train_list_coarse.txt', transform=train_transforms)
# test_dataset = CifarDataset(root_dir=train_dir, ann_file = '/tmp/pycharm_project_482/CIFAR100/CIFAR100/test_list_coarse.txt', transform=test_transforms)
#
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

class CaltechDataset(Dataset):
    def __init__(self, root_dir,  transform=None):

        self.root_dir = root_dir
        self.img_data,self.img_label,self.img_name = self.get_data()
        self.transform = transform
        print(len(self.img_data)==len(self.img_label))

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_data[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转化为RGB
        image = cv2.resize(image, (224,224),)
        # image = Image.open(self.img_data[idx])
        # print(image)
        label=self.img_label[idx]
        label=torch.from_numpy(label)
        # print(label)
        if self.transform:
            image = self.transform(image)
        return image,label

    def get_data(self):
        data = []
        labels = []
        labels_name = []
        i = 0
        file_paths=self.root_dir
        list_file = os.listdir(file_paths)
        for image_paths in list_file:
            imageDir = os.path.join(file_paths, image_paths)
            # print(image_paths)
            labels_name.append(image_paths)
            imageFile = os.listdir(imageDir)

            for file in imageFile:
                fileDir = os.path.join(imageDir, file)
                # print(fileDir)

                # image = cv2.imread(fileDir)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转化为RGB
                data.append(fileDir)

                labels.append(np.array(i,dtype=np.int64))
            i = i + 1
        # data = np.array(data)
        return data, labels, labels_name



data_dir = "/tmp/pycharm_project_821/Caltech101/Caltech101/caltech101/101_ObjectCategories"

train_transforms=torchvision.transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
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
            [   transforms.ToPILImage(),
                transforms.Resize(224),
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
dataset = CaltechDataset(root_dir=data_dir, transform=train_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#用普通的vit模型，batchsize是256
#vit混合模型，batchsize是128
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

model_ft = vit_base_patch32_224()
fc_features = model_ft.head.in_features
#修改类别为10，重定义最后一层
model_ft.head=nn.Linear(fc_features,102)
# model_ft=nn.DataParallel(model_ft)
print(model_ft)
filename1="/tmp/pycharm_project_821/best1.pt"
# 加载之前训练好的权重参数
filename = "best2.pt"
checkpoint = torch.load(filename1)
best_acc = checkpoint["best_acc"]
model_ft.load_state_dict(checkpoint["state_dict"])


# model_ft.load_state_dict(checkpoint,False)
# model_ft.load_state_dict(checkpoint)



device = "cuda:0"
feature_extract = True
# 保存文件的名字

#
#
# if feature_extract:
#     for param in model_ft.parameters():
#         param.requires_grad = False
#
# #解冻
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 20)


# GPU还是CPU计算
model_ft = model_ft.to(device)



# # 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)


optimizer_ft = optim.Adam(params_to_update, lr=0.001)
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
            print(
                    "Time elapsed {:.0f}m {:.0f}s".format(
                        time_elapsed // 60, time_elapsed % 60
                    )
                )
            print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
        print()


# train(1)
test(3)

# for param in model_ft.parameters():
#     param.requires_grad = True
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
#加载之前训练好的权重参数
# checkpoint = torch.load(filename)
# best_acc = checkpoint["best_acc"]
# model_ft.load_state_dict(checkpoint["state_dict"])
#
# train(25)
# test(5)