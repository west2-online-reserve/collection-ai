import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.nn import functional as F
import time
import warnings
from torch.utils.data import Dataset, DataLoader
import cv2
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

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



data_dir = "D:\\ApythonProject\\pytorch01\Begin01\\vit2\\Caltech101\\caltech101\\101_ObjectCategories"

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[1, 1, 1],
        padding=[0, 1, 0],
        first=False,
    ) -> None:
        """

        :param in_channels:
        :param out_channels:
        """
        super(ResBlk, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride[0],
                padding=padding[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride[1],
                padding=padding[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * 4,
                kernel_size=1,
                stride=stride[2],
                padding=padding[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * 4),
        )
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride[1],
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, ResBlk, num_classes=20):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # conv2
        self.conv2 = self._make_layer(ResBlk, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(
            ResBlk, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4
        )

        # conv4
        self.conv4 = self._make_layer(
            ResBlk, 256, [[1, 2, 1]] + [[1, 1, 1]] * 8, [[0, 1, 0]] * 9
        )



    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0, len(strides)):
            layers.append(
                block(
                    self.in_channels, out_channels, strides[i], paddings[i], first=flag
                )
            )
            flag = False
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


model_ft = ResNet50(ResBlk)

model_ft=ResNet50(ResBlk)
device = "cuda:0"
model_ft.to(device)


for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_ft(inputs)
                print(outputs.shape)