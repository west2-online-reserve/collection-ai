import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms#获取图像，图像数据预处理
from torch.utils.data import DataLoader

BATCH_SIZE=64
DEVICE = torch.device("cuda")
EPOCHS = 10

func_pic=transforms.Compose([transforms.ToTensor(),#变tensor并且除以255归一化
                            transforms.Normalize(mean=(0.1307,),std=(0.3081,))#官方提供的数据。N:减去均值，除以方差，使模型更快收敛
                            ])
#下载数据
train_set=datasets.MNIST("data",train=True,download=True,transform=func_pic)
test_set=datasets.MNIST("data",train=False,download=True,transform=func_pic)#下载存于root目录下

#加载数据
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

class primary(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1=nn.Linear(28*28,1000)
        self.f2=nn.Linear(1000,500)
        self.f3=nn.Linear(500,10)
        
        
    def forward(self,x):
        input_size=x.size(0)
        x=x.view(input_size, 784)
        x=self.f1(x)
        x=F.relu(x)
        x=self.f2(x)
        x=F.relu(x)
        x=self.f3(x)
        
        output=F.log_softmax(x,dim=1)
        return output
    


class module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,5)
        self.conv2=nn.Conv2d(10,20,3)
        self.fc1=nn.Linear(20*10*10,1000)
        self.fc2=nn.Linear(1000,10)
        
    def forward(self,x):
        input_size=x.size(0)
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)
        
        x=self.conv2(x)
        x=F.relu(x)
        
        x=x.view(input_size,-1)
        
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        
        output=F.log_softmax(x,dim=1)#softmax加负对数处理，越接近零越接近100%，做为反馈，训练使其尽可能接近零
        return output
    
    
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data.shape)

#创建模型及优化
model=primary().to(DEVICE)
optimizer=optim.Adam(model.parameters())

def train_model(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):#批次的一个（图片，标签)
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)#多分类计算损失函数
        loss.backward()#反向传播 ####用来计算梯度
        optimizer.step()#参数优化##用来算出新的参数，都是反向传播的环节
        if batch_index %3000 == 0:
            print("Train Epoch: {} \t Loss:{:.6f}".format(epoch,loss.item()))
            
def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    Accuracy=0.0
    test_loss=0.0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,target).item()#计算损失之和
            pred=output.argmax(dim=1)#概率最大的下标
            correct+=pred.eq(target.view_as(pred)).sum().item()#累计正确次数
        test_loss/=len(test_loader.dataset)
        Accuracy=100.0*correct/len(test_loader.dataset)#正确率
        print("Test_Average loss:{:4f},Accuracy:{:.3f}\n".format(test_loss,Accuracy))
        
        
for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)
torch.save(model.state_dict(),'model.ckpt')