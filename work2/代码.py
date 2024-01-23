%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data
import numpy
batchsize=16
d2l.use_svg_display()
print(torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cifar_path=r"C:\Users\20937\miniconda3\envs\python_3.9\data_cifar_100"
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers= 2

cifar_training=torchvision.datasets.CIFAR100(root=cifar_path,train=True,download=True,transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转（-10到10度之间）
    transforms.RandomCrop(32, padding=4),#随机裁剪   ##数据增强，有效减少模型拟合度过高
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
]))


cifar_testing=torchvision.datasets.CIFAR100(root=cifar_path,train=False,download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
]))

trainloader=torch.utils.data.DataLoader(cifar_training,batch_size=batchsize,shuffle=True,num_workers=num_workers)
testloader=torch.utils.data.DataLoader(cifar_testing,batch_size=batchsize,shuffle=False,num_workers=num_workers)

print("fine")


categories = {
    0: ['castle', 'skyscraper', 'bridge', 'house', 'road'],   
    1: ['beaver', 'dolphin', 'otter', 'seal', 'whale'],  
    2: ['bear', 'lion', 'tiger', 'wolf', 'leopard'], 
    3: ['beetle', 'bee', 'butterfly', 'caterpillar', 'cockroach'], 
    4: ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],   
    5: ['cloud', 'forest', 'mountain', 'plain', 'sea'], 
    6: ['cattle', 'chimpanzee', 'camel', 'elephant', 'kangaroo'],
    7: ['maple_tree', 'oak_tree', 'pine_tree', 'palm_tree', 'willow_tree'], 
    8: ['television', 'clock', 'keyboard', 'telephone', 'lamp'],
    9: ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    10: ['worm', 'spider', 'snail', 'lobster', 'crab'],
    11: ['plate', 'cup', 'bottle', 'bowl', 'can'],
    12: ['woman', 'baby', 'boy', 'girl', 'man'],
    13: ['skunk', 'fox', 'raccoon', 'possum', 'porcupine'],
    14: ['bus', 'pickup_truck', 'train', 'motorcycle', 'bicycle'], 
    15: ['aquarium_fish', 'ray', 'flatfish', 'shark', 'trout'],
    16: ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    17: ['tank', 'tractor', 'streetcar', 'rocket', 'lawn_mower'], 
    18: ['couch', 'chair', 'bed', 'wardrobe', 'table'],
    19: ['lizard','snake', 'turtle', 'dinosaur', 'crocodile']
}

def trasf():
    for i in range(1, 50000):     
        name = cifar_training.classes[cifar_training.targets[i]]

        if name in categories[0]:
            cifar_training.targets[i] = 0
        elif name in categories[1]:
            cifar_training.targets[i] = 1
        elif name in categories[2]:
            cifar_training.targets[i] = 2
        elif name in categories[3]:
            cifar_training.targets[i] = 3
        elif name in categories[4]:
            cifar_training.targets[i] = 4
        elif name in categories[5]:
            cifar_training.targets[i] = 5
        elif name in categories[6]:
            cifar_training.targets[i] = 6
        elif name in categories[7]:
            cifar_training.targets[i] = 7
        elif name in categories[8]:
            cifar_training.targets[i] = 8
        elif name in categories[9]:
            cifar_training.targets[i] = 9
        elif name in categories[10]:
            cifar_training.targets[i] = 10
        elif name in categories[11]:
            cifar_training.targets[i] = 11
        elif name in categories[12]:
            cifar_training.targets[i] = 12
        elif name in categories[13]:
            cifar_training.targets[i] = 13
        elif name in categories[14]:
            cifar_training.targets[i] = 14
        elif name in categories[15]:
            cifar_training.targets[i] = 15
        elif name in categories[16]:
            cifar_training.targets[i] = 16
        elif name in categories[17]:
            cifar_training.targets[i] = 17
        elif name in categories[18]:
            cifar_training.targets[i] = 18
        elif name in categories[19]:
            cifar_training.targets[i] = 19
    print('fine')

def trasff():
    for i in range(1, 10000):     
        name = cifar_testing.classes[cifar_testing.targets[i]]

        if name in categories[0]:
            cifar_testing.targets[i] = 0
        elif name in categories[1]:
            cifar_testing.targets[i] = 1
        elif name in categories[2]:
            cifar_testing.targets[i] = 2
        elif name in categories[3]:
            cifar_testing.targets[i] = 3
        elif name in categories[4]:
            cifar_testing.targets[i] = 4
        elif name in categories[5]:
            cifar_testing.targets[i] = 5
        elif name in categories[6]:
            cifar_testing.targets[i] = 6
        elif name in categories[7]:
            cifar_testing.targets[i] = 7
        elif name in categories[8]:
            cifar_testing.targets[i] = 8
        elif name in categories[9]:
            cifar_testing.targets[i] = 9
        elif name in categories[10]:
            cifar_testing.targets[i] = 10
        elif name in categories[11]:
            cifar_testing.targets[i] = 11
        elif name in categories[12]:
            cifar_testing.targets[i] = 12
        elif name in categories[13]:
            cifar_testing.targets[i] = 13
        elif name in categories[14]:
            cifar_testing.targets[i] = 14
        elif name in categories[15]:
            cifar_testing.targets[i] = 15
        elif name in categories[16]:
            cifar_testing.targets[i] = 16
        elif name in categories[17]:
            cifar_testing.targets[i] = 17
        elif name in categories[18]:
            cifar_testing.targets[i] = 18
        elif name in categories[19]:
            cifar_testing.targets[i] = 19
    print('fine')
trasf()##转换标签：细->粗
trasff()

#搭建算法模型32*32*3
DEVICE=torch.device("cuda")
epoch=5


class module(nn.Module):#20times->65%  #45times->70%#正常卷积神经网络
    def __init__(self):
        super().__init__()
        # self.convpre=nn.Conv2d(3,16,1)
        
        self.conv1=nn.Conv2d(3,16,3)
        self.conv2=nn.Conv2d(16,32,3)#->28

        self.conv3=nn.Conv2d(32,64,3)#14->
        self.conv4=nn.Conv2d(64,64,3)
        self.conv5=nn.Conv2d(64,64,3)#->8
        
        self.line1=nn.Linear(4*4*64,1000)
        self.line2=nn.Linear(1000,20)
        # self.line3=nn.Linear(1000,100)

        self.bn2=nn.BatchNorm2d(num_features=32)
        self.bn3=nn.BatchNorm2d(num_features=64)
        self.bn6=nn.BatchNorm1d(num_features=1000)
        self.bn1=nn.BatchNorm2d(num_features=16)
        self.bn4=nn.BatchNorm2d(num_features=64)
        self.bn5=nn.BatchNorm2d(num_features=64)

    def forward(self,x):
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=self.bn4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=self.bn5(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)

        x=x.view(batchsize,-1)

        x=self.line1(x)
        x=self.bn6(x)
        x=F.relu(x)
        x=self.line2(x)
        # x=self.bn4(x)
        # x=F.relu(x)
        # x=self.line3(x)

        return F.log_softmax(x,dim=1)



class resnet(nn.Module):#25times
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,16,5)#->28
        
        self.conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.expansion=nn.Conv2d(16,32,1)##1*1卷积

        self.line1=nn.Linear(14*14*32,2000)
        self.line2=nn.Linear(2000,20)
        
        self.bn1=nn.BatchNorm2d(num_features=32)
        self.bn2=nn.BatchNorm2d(num_features=32)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.bn4=nn.BatchNorm2d(num_features=16)
        self.bn5=nn.BatchNorm1d(num_features=2000)
        self.bn6=nn.BatchNorm2d(num_features=32)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn4(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)

        residual=x#16*14*14
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)#->32*14*14

        residual=self.expansion(residual)
        residual=self.bn6(residual)
        x+=residual
        x=F.relu(x)
        
        x=x.view(batchsize,-1)
        x=self.line1(x)
        x=self.bn5(x)
        x=F.relu(x)
        x=self.line2(x)

        return F.log_softmax(x,dim=1)

class GRU(nn.Module):#60times#卷积加上gru层的模型
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size
    ):
        super(GRU,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.conv1=nn.Conv2d(3,16,3)
        self.conv2=nn.Conv2d(16,32,3)#->28

        self.conv3=nn.Conv2d(32,64,3)#14->
        self.conv4=nn.Conv2d(64,64,3)
        self.conv5=nn.Conv2d(64,64,3)#->8
        
        self.line1=nn.Linear(4*4*64,2000)
        self.line2=nn.Linear(2000,20)
        # self.line3=nn.Linear(1000,100)

        self.bn2=nn.BatchNorm2d(num_features=32)
        self.bn3=nn.BatchNorm2d(num_features=64)
        self.bn6=nn.BatchNorm1d(num_features=2000)
        self.bn1=nn.BatchNorm2d(num_features=16)
        self.bn4=nn.BatchNorm2d(num_features=64)
        self.bn5=nn.BatchNorm2d(num_features=64)

        self.gru=nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.line=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=self.bn4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=self.bn5(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)

        x=x.view(batchsize,-1)

        x=self.line1(x)
        x=self.bn6(x)
        x=F.relu(x)
        x=self.line2(x)
        
        output,hide=self.gru(x)
        output=self.line(output)
        return F.log_softmax(output,dim=1)
        

model=GRU(20,64,3,20).to(DEVICE)
#model=module().to(DEVICE)
# model.load_state_dict(torch.load('cifar_100_model.ckpt'))
optimizer=optim.Adam(model.parameters())

def train_model(model,DEVICE,trainloader,optimizer,epoch):
    model.train()
    for index,(data,target) in enumerate(trainloader):
        
        data,target=data.to(DEVICE),target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)##批次中的loss均值
        loss.backward()
        optimizer.step()
        if index%2000==0:
            print("train epoch:",epoch)
            print(" loss:",loss.item())##提取张量为标量
for epo in range(1,epoch+1):
    train_model(model,DEVICE,trainloader,optimizer,epo)
    print("one")
    
torch.save(model.state_dict(),'cifar_100_model.ckpt')

#测试
def test_model(model,DEVICE,testloader):
    model.eval()
    ac=0.0
    correct=0
    with torch.no_grad():
        for index,(data,target) in enumerate(testloader):
            data,target=data.to(DEVICE),target.to(DEVICE)
            output=model(data)
            accu=output.argmax(dim=1)
            correct+=accu.eq(target.view_as(accu)).sum().item()
        ac=correct/len(testloader.dataset)
    print("正确率:",ac)
print('训练集：')
test_model(model,DEVICE,trainloader)
print('测试集：')
test_model(model,DEVICE,testloader)