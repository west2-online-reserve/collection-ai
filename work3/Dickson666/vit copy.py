import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image
import random
import math
from einops.layers.torch import Rearrange
from einops import rearrange
from torchsummary import summary

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

if not os.path.exists("./models"):
    os.makedirs("./models")

batch_size = 64
epoch = 100
patch_size = 16
learning_rate = 9e-5

transfer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5459, 0.5288, 0.5022), (0.3136, 0.3124, 0.3233))
])

dct = {}

class MyDataset(Dataset):

    def __init__(self, dir, transform) -> None:
        # print(dir)
        super().__init__()
        self.dir = str(dir),
        # print(os.listdir(dir))
        filenames = [f for f in os.listdir(dir)]
        self.lab = []
        self.dict = {}
        self.filenames = []
        for id, i in enumerate(filenames):
            dirr = os.path.join(dir, i)
            self.dict[i] = id
            dct[id] = i
            for j in os.listdir(dirr):
                self.lab.append(i)
                self.filenames.append(j)
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)

    # def getlab(self, dir):
    #     return os.path.dirname(dir)
    
    def __getitem__(self, index) -> torch.tensor:
        # print(type(self.dir), type(self.lab[index]), type(self.filenames[index]))
        dir = self.dir[0] + "/" + self.lab[index] + "/" + self.filenames[index]
        # print(dir)
        data = Image.open(dir).convert("RGB")
        if(self.transform):
            data = self.transform(data)
        return data, self.dict[self.lab[index]]
        
dataset = MyDataset('./data/caltech101/101_ObjectCategories', transfer)

# print(len(dataset))
# exit(0)

num = [i for i in range(len(dataset))]

random.shuffle(num)

train_len = int(len(dataset) * 0.7)

# test = random.randint(0, len(dataset) - 1)
# test_img, test_lab = dataset[test]
# test_img = test_img.permute(1, 2, 0)
# plt.imshow(test_img)
# plt.title(dct[test_lab])
# plt.show()

train_dataset = []
test_dataset = []

for i in range(train_len):
    train_dataset.append(dataset[num[i]])
for i in range(train_len, len(num)):
    test_dataset.append(dataset[num[i]])

print(len(train_dataset), len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


# exit(0)


class Pre_work(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, 784))
        self.emd = nn.Parameter(torch.randn(1, 197, 784))
        self.linear = nn.Linear(784, 768)
        self.covn = nn.Sequential(
            # Rearrange('b e (h p1) (w p2) -> b (h w) (p1 p2 e)', p1 = 16, p2 = 16),
            # nn.Linear(768, 768)
            nn.Conv2d(3, 196, 8, 8),
            nn.Flatten(2),
            nn.Linear(784, 784)
        )
    
    def forward(self, x):
        '''
        # print(x.shape)
        x = x.view(x.shape[0], 3, 224, -1, patch_size)
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        # print(x.shape)
        x = x.view(x.shape[0], 3, x.shape[2], x.shape[3], -1, patch_size)
        # print(x.shape)
        x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = x.view(x.shape[0], -1, x.shape[3])
        # print(x.shape, "/n", self.emd.shape)
        # print(x.device, self.emd.device)
        '''
        x = self.covn(x)
        # print(x.shape)
        return self.linear(torch.cat([self.cls.expand(x.shape[0], 1, self.cls.shape[2]), x], dim = 1) + self.emd)#注意顺序

def trans_pos(x, n):
    # x = x.view(x.shape[0], 197, n, -1)
    # x = x.permute(0, 2, 1, 3).contiguous()
    # x = x.reshape(-1, x.shape[2], x.shape[3])
    x = rearrange(x, 'b n (h d) -> b h n d', h = n)
    return x
def trans_neg(x, n):
    # x = x.view(n, -1, x.shape[1], x.shape[2])
    # x = x.permute(0, 2, 3, 1).contiguous()
    # x = x.reshape(x.shape[0], x.shape[1], -1)
    x = rearrange(x, 'b h n d -> b n (h d)')
    return x

class dotproductattention(nn.Module):
    def __init__(self, dropout = 0.15) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, q, v, k):
        d = q.shape[-1]
        res = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        res = nn.functional.softmax(res, dim = -1)
        return torch.matmul(self.drop(res), v)

class multihead(nn.Module):
    def __init__(self, q_size, v_size, k_size, head_num, hide_num) -> None:
        super().__init__()
        self.attention = dotproductattention()
        self.W_q = nn.Linear(q_size, hide_num)
        self.W_v = nn.Linear(v_size, hide_num)
        self.W_k = nn.Linear(k_size, hide_num)
        self.W_o = nn.Linear(hide_num, q_size)
        self.head_num = head_num
        
    def forward(self, q, v, k):
        Q = trans_pos(self.W_q(q), self.head_num)
        K = trans_pos(self.W_k(k), self.head_num)
        V = trans_pos(self.W_v(v), self.head_num)
        res = self.attention(Q, V, K)
        # print("REs:", res.shape)
        return self.W_o(trans_neg(res, q.shape[0]))

class vit_block(nn.Module):
    def __init__(self, n, dropout = 0.2) -> None:
        super().__init__()
        self.multihead = multihead(768, 768, 768, 12, 1536)
        # self.multihead = nn.MultiheadAttention(768, 12)
        self.l1 = nn.LayerNorm((197, 768))
        self.l2 = nn.LayerNorm((197, 768))
        self.l3 = nn.LayerNorm((197, 768))
        self.mlp = nn.Sequential(
            nn.Linear(768, n),
            nn.ReLU(),
            # nn.BatchNorm1d(197),
            nn.Linear(n, 768)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        y = self.l1(x)
        y = self.multihead(y, y, y)
        y = self.dropout(y)
        y = y + x
        z = self.l2(y)
        z = self.mlp(z)
        z = self.dropout(z)
        z = z + y
        return self.l3(z)

class vit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = Pre_work()
        self.v1 = vit_block(2048, 0.)
        self.v2 = vit_block(2048, 0.)
        self.v3 = vit_block(2048, 0.)
        self.v4 = vit_block(2048, 0.1)
        self.v5 = vit_block(2048, 0.1)
        self.v6 = vit_block(2048, 0.5)
        self.linear = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 101)
        )
        self.drop = nn.Dropout(0.1)
        self.id = nn.Identity()
    
    def forward(self, x, epoch = 0):
        x = self.pre(x)
        x = self.drop(x)
        x = self.v2(self.v1(x))
        x = self.v3(x)
        x = self.v4(x)
        x = self.v5(x)
        x = self.v6(x)
        # x = self.v7(x)
        # x = self.v8(x)
        if(False):
            output = x.to("cpu").numpy()[0]
            output = (output - np.min(output)) / (np.max(output) - np.min(output))
            plt.figure(figsize=(10, 20)) 
            plt.imshow(output, cmap='viridis')  
            plt.colorbar()  
            plt.title('Heatmap')
            plt.savefig("./Heatmap/"+str(epoch) + " epoch Heatmap.png")
            plt.clf()
        x = self.id(x[:, 0, :])
        return self.linear(torch.squeeze(x))

model = vit().to(device)
# summary(model, (3, 224, 224), batch_size=batch_size, device=device)

optims = optim.Adam(model.parameters(), lr = learning_rate)
crit = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optims, step_size = 1, gamma = 0.87)

std_loss = range(10)

Tl = []
Epoch = []
Loss = []
Acc = []
train_loss = []

def train(dataloader, model, ep):
    model.train()
    for i, (image, label) in enumerate(dataloader):
        # print("QWQ")
        # print(image, label)
        # print(label)
        image = image.to(device)
        label = label.to(device)
        # if(image.shape[0] != batch_size):
        #     continue
        optims.zero_grad()
        res = model(image)
        loss = crit(res, label)
        loss.backward()
        optims.step()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss :>7f} lr: {optims.param_groups[0]["lr"]}')
            train_loss.append(loss)
            Epoch.append(ep + (i / len(dataloader)))

def test(dataloader, model, ep):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            res = model(img, ep)
            loss = crit(res, label)
            test_loss += loss.item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    correct /= len(dataloader.dataset)
    test_loss /= len(dataloader)
    Loss.append(test_loss)
    Acc.append(correct)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')

EPoch = [0]
Tl = []
std_loss = range(10)
std_loss = [ss / 2.0 for ss in std_loss]


if __name__ == "__main__":
    for i in range(epoch):
        train(train_dataloader, model, i)
        test(test_dataloader, model, i + 1)
        EPoch.append(i + 1)
        plt.title("VIT " + str(i + 1) + " Epoch")
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
        plt.savefig("./fig/" + str(i + 1) + "Epoch.png")
        plt.clf()
        scheduler.step()