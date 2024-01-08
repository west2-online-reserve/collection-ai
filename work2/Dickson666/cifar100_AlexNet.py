#CNN with AlexNet model

#Written by Dickson

#Started on 12/10/2023

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

if not os.path.exists("./models"):
    os.makedirs("./models")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Running on " + device + ".......")

batch_size = 64
learning_rate = 6e-5
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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.works = nn.Sequential(# 3 * 224 * 224
            nn.Conv2d(3, 96, 11, 4, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 384, 5, padding = 2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(384, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 384, 3, padding = 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 100)
        )
    def forward(self, x):
        return self.works(x)

model = CNN().to(device)
# summary(model, (3, 224, 224), device="cpu")
if os.path.exists("modlog"):
    mod_idx = open("modlog", "r").read()
    model.load_state_dict(torch.load("./models/model_" + mod_idx + ".pth"))
crit = nn.CrossEntropyLoss()
optims = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optims, step_size = 30, gamma = 0.5)

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
# 33 % max, fail