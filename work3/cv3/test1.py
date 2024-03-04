
import cv2
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
from vit_hybrid import vit_base_patch32_224
from torch.nn import functional as F

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image
from resnet import ResNet50,ResBlk

class CaltechDataset(Dataset):
    def __init__(self, root_dir,  transform=None):

        self.root_dir = root_dir
        self.img_data,self.img_label,self.img_name = self.get_data()
        self.transform = transform
        print(len(self.img_data)==len(self.img_label))
        print(len(self.img_name))

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
        i = -1
        file_paths=self.root_dir
        list_file = os.listdir(file_paths)
        for image_paths in list_file:
            imageDir = os.path.join(file_paths, image_paths)
            # print(image_paths)
            labels_name.append(image_paths)
            imageFile = os.listdir(imageDir)
            i = i + 1

            for file in imageFile:
                fileDir = os.path.join(imageDir, file)
                # print(fileDir)

                # image = cv2.imread(fileDir)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转化为RGB
                data.append(fileDir)

                labels.append(np.array(i,dtype=np.int64))

        # data = np.array(data)
        return data, labels, labels_name



data_dir = "D:\\ApythonProject\\pytorch01\Begin01\\vit\\Caltech101\\caltech101\\101_ObjectCategories"

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = "cuda:0"
model_ft=ResNet50(ResBlk)
model_ft.to(device)


for idx,(x,y)in enumerate(test_loader):
    model_ft.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ft(inputs)
            print(outputs.shape)
