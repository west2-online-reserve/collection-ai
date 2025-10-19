import datetime
import argparse


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter



import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader

import time
import warnings
from torch import nn
import cv2
from torch.nn import functional as F
from utils import *
from utils1 import *
from train import *
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image



model_ft =Model()
print(model_ft)
filename = "best4.pt"
filename1 = "best5.pt"
filename3='best6.pt'
filename2="yolov5s1.pt"
# train_path="D:\ApythonProject\pytorch01\Begin01\yolov3_spp\data\my_train_data.txt"
# test_path="D:\ApythonProject\pytorch01\Begin01\yolov3_spp\data\my_val_data.txt"
imgsz=640
batch_size=64
# hyp=check_file("D:\ApythonProject\pytorch01\Begin01\yolo11\models\hyp.scratch.yaml")
device = "cuda:0"
accumulate=max(round(64 / batch_size), 1)
hyp={'box': 0.05,  # box loss gain
'cls': 0.5,  # cls loss gain
'cls_pw': 1.0,  # cls BCELoss positive_weight
'obj': 1.0,  # obj loss gain (scale with pixels)
'obj_pw': 1.0,  # obj BCELoss positive_weight
'iou_t': 0.20 , # IoU training threshold
'anchor_t': 4.0,  # anchor-multiple threshold
 'anchors': 3,  # anchors per output layer (0 to ignore)
'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
'hsv_h': 0.015 , # image HSV-Hue augmentation (fraction)
'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
'hsv_v': 0.4,
'degrees': 0.0,
     'translate': 0.1,  # image translation (+/- fraction)
     'scale': 0.5,  # image scale (+/- gain)
     'shear': 0.0,  # image shear (+/- deg)
     'perspective': 0.0
     }

data='/tmp/pycharm_project_508/models/model1/data/my_data.data'
data_dict = parse_data_cfg(data)
train_path = data_dict["train"]
test_path = data_dict["valid"]
nc = int(data_dict["classes"])  # number of classes
nl = 3
print(hyp['box'])
hyp['box'] *= 3. / nl  # scale to layers
hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers


train_dataset = LoadImagesAndLabels(train_path, imgsz, batch_size,
                                        augment=True,
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=True,  # rectangular training
                                        cache_images=False,
                                        single_cls=False)

    # 验证集的图像尺寸指定为img_size(512)
val_dataset = LoadImagesAndLabels(test_path, imgsz, batch_size,
                                      hyp=hyp,
                                      rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                      cache_images=False,
                                      single_cls=False)


# val_size = int(0.5 * len(val_dataset))
# val_dataset1, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, len(val_dataset)-val_size])
mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
assert mlc < 20, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc)
# dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=0,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)


model_ft.nc = 20  # attach number of classes to model
model_ft.hyp = hyp  # attach hyperparameters to model
model_ft.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
# fc_features = model_ft.head.in_features
# #修改类别为10，重定义最后一层
# model_ft.head=nn.Linear(fc_features,102)
# model_ft=nn.DataParallel(model_ft)
# print(model_ft)
# filename1="/tmp/pycharm_project_821/best1.pt"
# 加载之前训练好的权重参数

checkpoint = torch.load(filename3)
best_acc = checkpoint["best_acc"]
model_ft.load_state_dict(checkpoint["state_dict"])


# model_ft.load_state_dict(checkpoint,False)
# model_ft.load_state_dict(checkpoint)

# GPU还是CPU计算
model_ft = model_ft.to(device)

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数


def runn1():
 for epoch in range(1):
    best_acc=0


    mp, mr, map50, map = evaluate1(model_ft, val_datasetloader, device=device)
    # test(3)

    print("mp:{},mr:{},map50:{},map:{}".format(mp, mr, map50, map))

    state = {
            "state_dict": model_ft.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
            "best_acc": best_acc,
            "optimizer": optimizer_ft.state_dict(),
        }
    torch.save(state, filename1)
#
# runn1()

epoches=50
loss1=2.19

mp, mr, map50, map = evaluate1(model_ft, train_dataloader,device=device)
print("mp:{},mr:{},map50:{},map:{}".format(mp, mr, map50, map))

for epoch in range(epoches):

    print("Epoch {}/{}".format(epoch, epoches))
    checkpoint = torch.load(filename3)
    model_ft.load_state_dict(checkpoint["state_dict"])

    best_acc=0
    # mloss = torch.zeros(4, device=device)  # mean losses
    mloss,loss=train(model_ft,optimizer_ft,train_dataloader,accumulate)
    if(loss1>loss):
        loss1=loss
        state = {
            "state_dict": model_ft.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
            "best_acc": best_acc,
            "optimizer": optimizer_ft.state_dict(),
        }
        torch.save(state, filename3)
        print("update loss:{}".format(loss1))

    # test(3)


    print(mloss,loss)
    if (epoch % 5 == 0):
        print("evaluate")
        # print("train:")
        # mp, mr, map50, map = evaluate1(model_ft, train_dataloader, coco=coco, device=device)
        # print("mp:{},mr:{},map50:{},map:{}".format(mp, mr, map50, map))

        print("test:")
        runn1()





