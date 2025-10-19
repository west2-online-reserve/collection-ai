#CNN with LeNet model

#Written by Dickson

#Started on 10/10/2023

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchsummary import summary

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
learning_rate = 1e-3
epoch = 10
save_epoch = 10
models_remain = 3

transfer = transforms.Compose([
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
        self.works = nn.Sequential(# 3 * 32 * 32
            nn.Conv2d(3, 6, 5, padding = 2), # 6 * 32 * 32
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),# 6 * 16 * 16
            nn.Conv2d(6, 16, 5),# 16 * 12 * 12
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),# 16 * 6 * 6
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 100)
        )
    def forward(self, x):
        return self.works(x)

model = CNN().to(device)
# summary(model, (3, 32, 32), device="cpu")
# if os.path.exists("modlog"):
#     mod_idx = open("modlog", "r").read()
#     model.load_state_dict(torch.load("./models/model_" + mod_idx + ".pth"))
crit = nn.CrossEntropyLoss()
optims = optim.Adam(model.parameters(), lr = learning_rate)

def train(dataloader, model, ep):
    model.train()
    for i, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        # print(image, label)
        optims.zero_grad()
        res = model(image)
        loss = crit(res, label)
        loss.backward()
        optims.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss :>7f}')

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
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')

for i in range(epoch):
    train(training_dataloader, model, i)
    test(test_dataloader, model)
    # if (i + 1) % save_epoch == 0:
        # torch.save(model.state_dict(), "./models/model_" + str(i + 1) + ".pth")
        # open("modlog", "w").write(str(i + 1))
        # print(f'Saveing model_{i + 1}.pth...')
        # if (i + 1) - save_epoch * models_remain > 0 and os.path.exists("./models/model_" + str((i + 1) - save_epoch * models_remain) + ".pth"):
        #     os.remove("./models/model_" + str((i + 1) - save_epoch * models_remain) + ".pth")

# 33 % max, fail