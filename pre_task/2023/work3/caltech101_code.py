import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import numpy

batchsize=2
depth=6
epoch=10
DEVICE=torch.device("cuda")

print(torch.cuda.is_available())
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\20937\Downloads\101_ObjectCategories', transform=transform)
traindataset, testdataset = data.random_split(dataset, [6858, 2286])

print('yes')

#强化集
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转（-10到10度之间）
    transforms.RandomCrop(32, padding=4),#随机裁剪   ##数据增强，有效减少模型拟合度过高
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batchsize, shuffle=False)
print("fine")

#普通集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batchsize, shuffle=True)
print("fine")




#搭建vit模型(3,224,224)
class patch_class_Embedding(nn.Module):#3->0.15,6->0.3
    def __init__(self):
        super().__init__()
        
        # self.conv=nn.Conv2d(3,768,16,16)
        self.conv1=nn.Conv2d(3,768,2,2)
        self.conv2=nn.Conv2d(768,768,2,2)
        self.conv3=nn.Conv2d(768,768,2,2)
        self.conv4=nn.Conv2d(768,768,2,2)

        self.class_token=nn.Parameter(torch.zeros(batchsize,1,768))
        self.position=nn.Parameter(torch.zeros(batchsize,197,768))

        self.drop=nn.Dropout(p=0.2)
        self.norm=nn.LayerNorm(768)
        
    def forward(self,x):
        # x=self.conv(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        
        x=x.view(batchsize,768,-1).transpose(1,2)##->(b,196,768)

        x=torch.cat((self.class_token,x),dim=1)##->(b,197,768)###

        x=x+self.position

        x=self.drop(x)
        return self.norm(x)
##transformer encoder###
##别忘了x与multi——attn与mlp的残差链接结构
class multi_attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(768,3*768)
        self.scale=(768//12)**-0.5##num_heads=12
        self.drop=nn.Dropout(p=0.2)
        # self.line=nn.Linear(768,768)
    def forward(self,x):
        qkv=self.qkv(x).reshape(batchsize,197,3,12,64).permute(2,0,3,1,4)##->(3,b,12,197,64)
        q,k,v=qkv[0],qkv[1],qkv[2]

        attn=(q@k.transpose(-2,-1))*self.scale##->(b,12,197,197)
        attn=attn.softmax(dim=-1)

        attn=(attn@v).transpose(1,2).reshape(batchsize,197,768)
        
        attn=self.drop(attn)
        
        return attn+x
class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.line1=nn.Linear(768,4*768)
        self.line2=nn.Linear(4*768,768)
        self.norm=nn.LayerNorm(768)

        self.drop=nn.Dropout(p=0.2)

    def forward(self,x):
        y=self.norm(x)
        y=self.line1(y)
        y=F.gelu(y)
        y=self.line2(y)
        y=self.drop(y)
        return x+y##->(b,197,768)



class vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.line=nn.Linear(768,102)

        self.patch_embed=patch_class_Embedding()
        self.transformer_encoder=nn.Sequential(*[multi_attn(),Mlp()]*depth)

    def forward(self,x):
        x=self.patch_embed(x)
        x=self.transformer_encoder(x)

        x=x[:,0,:]        

        x=self.line(x)
        
        x=F.log_softmax(x,dim=-1)#
        return x.reshape(batchsize,102)
        
        

model=vit().to(DEVICE)
# model.load_state_dict(torch.load('caltech10112.ckpt'))#x20，40，0.44，0.99，无效强化训练
# model.load_state_dict(torch.load('caltech10111.ckpt'))#0.68
# model.load_state_dict(torch.load('caltech10113.ckpt'))#30，x10
# model.load_state_dict(torch.load('caltech10114.ckpt'))#30，x20，5，x5，5--0.88
# model.load_state_dict(torch.load('caltech10115.ckpt'))#30，x20，5，x5，5，x5
# model.load_state_dict(torch.load('caltech10110.ckpt'))#30，x20，10

# optimizer=optim.Adam(model.parameters(),lr=0.00001)
optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=epoch)#余弦动态lr


def test_model(model,DEVICE,testloader):#0.43-40epo-标准集
    model.eval()
    ac=0.0
    correct=0
    with torch.no_grad():
        for index,(data,target) in enumerate(testloader):
            data,target=data.to(DEVICE),target.to(DEVICE)
            output=model(data)
            accu=output.argmax(dim=1)
            correct+=accu.eq(target.view_as(accu)).sum().item()
    print("正确率:",correct/2286)
    

def train_model(model,DEVICE,trainloader,optimizer,epoch):
    model.train()
    for index,(data,target) in enumerate(trainloader):
        
        data,target=data.to(DEVICE),target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        
        loss=F.cross_entropy(output,target)##批次中的loss均值
        loss.backward()
        optimizer.step()
        if index%500==0:
            print("train epoch:",epoch)
            print(" loss:",loss.item())



for epo in range(1,epoch+1):
    train_model(model,DEVICE,trainloader,optimizer,epo)
    test_model(model,DEVICE,testloader)
    
    scheduler.step()
    print("one")
    
# torch.save(model.state_dict(),'caltech10115.ckpt')


