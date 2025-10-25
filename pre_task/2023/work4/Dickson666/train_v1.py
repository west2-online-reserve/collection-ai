from utils.input import read_train, read_val, cmp
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.YOLOv1 import Yolo
from tools import Criterion_v1
from trainer_v1 import train,test
import matplotlib.pyplot as plt
import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("device: ", device)

feature_size = [224, 224]

transfer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(feature_size)
])

batch_size = 16
lr = 7e-5
epoch = 100

print("Reading data....")
train_img, train_label, train_bbox = read_train()
test_img, test_label, test_bbox = read_val()
class Mydataset(Dataset):
    def __init__(self, img, label, bbox, transform):
        super().__init__()
        self.img = img
        self.label = label
        self.bbox = bbox
        self.transform = transform
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        bbox = self.bbox[idx]
        if(self.transform != None):
            img = self.transform(img)
        return img, label, bbox

train_Dataset = Mydataset(train_img, train_label, train_bbox, transfer)
test_Dataset = Mydataset(test_img, test_label, test_bbox, transfer)
train_Dataloader = DataLoader(train_Dataset, batch_size, shuffle=True, collate_fn = cmp)
test_Dataloader = DataLoader(test_Dataset, batch_size, shuffle=True, collate_fn = cmp)

print("Finished reading!")

model = Yolo().to(device)
model.load_state_dict(torch.load("./model saved/yolov1.pth"))
model.fc = nn.Sequential(
    nn.Conv2d(1024, 1024, 3, padding=1),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(1024, 1024, 3, padding=1),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(1024 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1470),
    # nn.Sigmoid()
).to(device)
# print(child)
optims = optim.SGD(model.parameters(), lr, 0.937, weight_decay = 5e-4)
# optims = optim.Adam(model.parameters(), lr)
scheduler = optim.lr_scheduler.StepLR(optims, step_size=1,gamma= 0.1)
crit = Criterion_v1()

print("Start training...")

ep1, ep2, los1, los2, mAP, reall = [], [], [], [], [], []

for i in range(epoch):
    ls, se = train(model, i, train_Dataloader, crit, optims, device, epoch, scheduler)
    los1 += ls
    ep1 += se
    ma, re, lss = test(model, test_Dataloader, crit, device)
    mAP.append(ma * 10)
    reall.append(re * 10)
    los2.append(lss)
    ep2.append(i + 1)
    if(i == 30 or i == 50 or i == 70):
        scheduler.step()
    # scheduler.step()
    if((i+1) % 10 == 0):
        torch.save(model.state_dict(), "./models/model_" + str(i + 1) + ".pth")
        plt.plot(ep1, los1, label = "train loss")
        plt.plot(ep2, mAP, label = "mAP (*10)")
        plt.plot(ep2, reall, label = "recall (*10)")
        plt.plot(ep2, los2, label = "test loss")
        plt.legend(loc = "best")
        plt.show()
