#CNN with ResNet model

#Written by Dickson

#Started on 16/10/2023

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import torch.nn.functional as F

if not os.path.exists("./cifar100/Logs"):
    os.makedirs("./cifar100/Logs")
open("./cifar100/Logs/log.txt", "w").write("")

if not os.path.exists("./models"):
    os.makedirs("./models")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# torch.cuda.set_per_process_memory_fraction(0.5)
print("Running on " + device + ".......")

batch_size = 64
learning_rate = 1e-3
epoch = 100
save_epoch = 5
models_remain = 3

transfer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
])

training_data = datasets.CIFAR100(
    root = "./data",
    train = True,
    transform= transfer,
    download=True
)

test_data = datasets.CIFAR100(
    root = "./data",
    train = False,
    transform = transfer,
    download = True
)

training_dataloader = DataLoader(dataset = training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, dif = False, strides = 1):
        super().__init__()
        self.work1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = 1, stride = strides, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.work2 = nn.Sequential()
        if dif:
            self.work2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = strides, bias=False),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.work1(x)
        res += self.work2(x)
        res = F.relu(res)
        return res

def ResNet_Block(in_channels, out_channels, block_num, first_block = False):
    res = []
    for i in range(block_num):
        if i == 0 and not first_block:
            res.append(Residual(in_channels, out_channels, dif=True, strides= 2))
        else:
            res.append(Residual(out_channels, out_channels))
    return res

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.b2 = nn.Sequential(*ResNet_Block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*ResNet_Block(64, 128, 2))
        self.b4 = nn.Sequential(*ResNet_Block(128, 256, 2))
        # self.b5 = nn.Sequential(*ResNet_Block(256, 512, 2))
        self.f1 = nn.AdaptiveAvgPool2d((1, 1))
            #nn.Flatten(),
        self.f2 = nn.Linear(256, 100)
    def forward(self, x):
        res = self.b1(x)
        res = self.b2(res)
        res = self.b3(res)
        res = self.b4(res)
        # res = self.b5(res)
        res = self.f1(res)
        res = res.reshape(x.shape[0], -1)
        res = self.f2(res)
        return res

model = CNN().to(device)
summary(model, (3, 224, 224), device=device)
if os.path.exists("modlog"):
    mod_idx = open("modlog", "r").read()
    model.load_state_dict(torch.load("./models/model_" + mod_idx + ".pth"))
crit = nn.CrossEntropyLoss()
optims = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optims, step_size = 10, gamma = 0.5)

train_loss=[]
Epoch=[]

def train(dataloader, model, ep):
    model.train()
    for i, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        # print(image.size())
        optims.zero_grad()
        res = model(image)
        loss = crit(res, label)
        loss.backward()
        optims.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss :>7f}')
            open("./cifar100/Logs/log.txt", "a").write(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{i + 1:>4d} / {len(dataloader) :>4d}] Loss: {loss :> 7f}\n')
            train_loss.append(loss)
            Epoch.append(ep + (i / len(dataloader)))

Loss = []
Acc = []
EPoch = [0]

def test(dataloader, model):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            res = model(img)
            loss = crit(res, label)
            test_loss += loss.item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    correct /= len(dataloader.dataset)
    test_loss /= len(dataloader)
    Loss.append(test_loss)
    Acc.append(correct)
    # Loss = np.append(Loss, test_loss)
    # Acc = np.append(Acc, correct)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')
    open("./cifar100/Logs/log.txt", "a").write(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')

std_loss = range(12)

std_loss = [ss / 2.0 for ss in std_loss]

Tl = []

for i in range(epoch):
    train(training_dataloader, model, i)
    test(test_dataloader, model)
    EPoch.append(i + 1)
    plt.title("CIFAR100 training on AlexNet " + str(i + 1) + " Epoch")
    plt.xticks(EPoch)
    plt.yticks(std_loss)
    plt.xlabel("EPOCH")
    plt.plot(EPoch[1:], Loss, label = 'test loss')
    plt.plot(EPoch[1:], Acc, label = 'Acc')
    # Epoch = [ep.cpu().numpy() for ep in Epoch]
    Tl = Tl + [tl.cpu().detach().numpy() for tl in train_loss]
    train_loss = []
    plt.plot(Epoch, Tl, label = 'train loss')
    plt.legend(loc='best')
    plt.savefig("./cifar100/Logs/AlexNet_" + str(i + 1) + "Epoch.png")
    plt.clf()
    # plt.show()
    if (i + 1) % save_epoch == 0:
        torch.save(model.state_dict(), "./models/model_" + str(i + 1) + ".pth")
        open("modlog", "w").write(str(i + 1))
        print(f'Saveing model_{i + 1}.pth...')
        if (i + 1) - save_epoch * models_remain > 0 and os.path.exists("./models/model_" + str((i + 1) - save_epoch * models_remain) + ".pth"):
            os.remove("./models/model_" + str((i + 1) - save_epoch * models_remain) + ".pth")
    scheduler.step()
