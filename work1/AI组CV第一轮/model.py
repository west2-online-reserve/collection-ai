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

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out

class ResNet18LSTM(nn.Module):

    def __init__(self, input_size=32*32, hidden_size=32*32, n_class=20,dim=1024):
        super(ResNet18LSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=4,dropout=0.8,batch_first=True)
        self.output_Layer = nn.Sequential(
            nn.Linear(1000, n_class),
            # nn.Softmax(dim=-1)
        )
        self.resnet=models.resnet18(pretrained=True)

    def forward(self, x):

        """
        :param x:
        :return:
        """
        # x = F.relu(self.conv1(x))
        # #
        # x = self.blk1(x)
        # x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)
        # print(x.shape)
        # x = torch.transpose(x, 0, 2)
        # print(x.shape)
        # x = torch.transpose(x, 1, 2).flatten(2)
        b, _, w, h = x.shape
        x = x.flatten(2)
        output,_ =self.lstm1(x)

        x= torch.reshape(x,(b, 3, w, h))
        x= self.resnet(x)
        x=self.output_Layer(x)


        # out = self.output_Layer(x)
        return x




