import torch
import os
import math
import random
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


epoch = 20
learning_rate = 2e-4
batch_size = 64
num_step = 8 # 一次处理的区间大小，t
# device = "cpu"

transfer = transforms.Compose([
    # transforms.Resize(32),
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

vocab_len = 32 * 8 * 4

num_hiddens = 1024 # 隐藏层大小
num_dir = 1
rnn_layer = nn.GRU(vocab_len, num_hiddens, num_dir, batch_first = True) # output: (sequence_length, batch_size, num_directions * hidden_size), hidden = (num_layers * num_directions, batch_size, hidden_size)
# nn.RNN输出并不是结果，而是最后一个隐藏层，所以最后一个维度为num_directions * hidden_size
# 初始化隐藏状态 ↓
state = torch.zeros((num_dir, num_step, num_hiddens)) 
# cell = torch.zeros((1, num_step, num_hiddens)) 

class RNN(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.b1 = nn.Sequential(
            # nn.Conv2d(3, 32, 7, 2, 3),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.flatten = nn.Flatten()
        if not self.rnn.bidirectional:
            self.num_dir = 1
            self.linear = nn.Linear(self.num_hiddens, 100)
        else:
            self.num_dir = 2
            self.linear = nn.Linear(self.num_hiddens * 2, 100)
    
    def forward(self, inputs, state):
        x = self.b1(inputs)
        # print(x.shape)
        # x = self.flatten(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,128 * 8)
        y, state = self.rnn(x)
        # print(y.shape)
        # y = self.flatten(y)
        output = self.linear(y[:, -1, :])
        return output, state

model = RNN(rnn_layer, vocab_len).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:", total_params)

Loss = []
Acc = []
max_Acc = 0.0

def test(model, dataloader):
    states = torch.zeros((num_dir, num_step, num_hiddens)).to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, label in dataloader:
            input, label = input.to(device), label.to(device)
            if(input.shape[0] != batch_size):
                continue
            # input = input_prework(input)
            res,_  = model(input, states)
            loss = crit(res, label)
            test_loss += loss.item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    correct /= len(dataloader.dataset)
    test_loss /= len(dataloader)
    Loss.append(test_loss)
    Acc.append(correct)
    # Loss = np.append(Loss, test_loss)
    # Acc = np.append(Acc, correct)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg perplexity:{test_loss :> 8f}')
    # open("./cifar100/Logs/log.txt", "a").write(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')
    return correct

# print(test(model, "Select your preferences and run the install command", 200))

# if not os.path.exists("./models/model.pth"):
if 1: 
    crit = nn.CrossEntropyLoss()
    optims = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optims, step_size = 1, gamma = 0.996)

    train_loss=[]
    Epoch=[]

    def train(model, dataloader, ep, states):
        model.train()
        states = states.to(device)
        for i, (input, label) in enumerate(dataloader):
            optims.zero_grad()
            input, label = input.to(device), label.to(device)
            if(input.shape[0] != batch_size):
                continue
            # input = input_prework(input)
            res,_ = model(input, states)
            # print(res.shape)
            loss = crit(res, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度剪裁
            optims.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] perplexity: {loss :>7f} lr: {optims.param_groups[0]["lr"] :>6f}')
                train_loss.append(loss)
                Epoch.append(ep + (i / len(dataloader)))
                # scheduler.step()

    EPoch = [0]
    Tl = []
    std_loss = range(100)

    # std_loss = [ss / 2.0 for ss in std_loss]

    for i in range(epoch):
        print(i, "QWQ")
        states = state
        train(model, training_dataloader, i, states)
        res = test(model, test_dataloader)
        max_Acc = max(res, max_Acc)
        print(f'Max_Acc{(max_Acc * 100) :> 0.1f} %\n')
        plt.title("RNN " + str(i + 1) + " Epoch")
        plt.xticks(EPoch)
        plt.yticks(std_loss)
        plt.xlabel("EPOCH")
        EPoch.append(i + 1)
        plt.plot(EPoch[1:], Loss, label = 'test loss')
        plt.plot(EPoch[1:], Acc, label = 'Acc')
        scheduler.step()
        Tl = Tl + [tl.cpu().detach().numpy() for tl in train_loss]
        train_loss = []
        plt.plot(Epoch, Tl, label = 'train loss')
        plt.legend(loc='best')
        plt.savefig("./cifar100/Logs/RNN_" + str(i + 1) + "Epoch.png")
        plt.clf()
    
    torch.save(model.state_dict(), "./models/model.pth")

else:
    model.load_state_dict(torch.load("./models/model.pth"))

print(test(model, "import torch.optim as optim", 200))