import torchvision
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
#下载数据集并读取
batch_size=256
def load_data(batch_size):
    trans=transforms.ToTensor()
    mnist_train=torchvision.datasets.MNIST(root="/data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.MNIST(root="/data",train=False,transform=trans,download=True)
    return(data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4),
           data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4))
train_iter,test_iter=load_data(batch_size)
#定义模型
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)
#定义损失函数
loss=nn.CrossEntropyLoss()
#定义优化器
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
#定义准确率
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    y_hat=y_hat.type(y.dtype)
    cmp=y_hat==y
    return float(cmp.type(y.dtype).sum()),float(y.shape[0])
#定义训练函数
def train_epoch(net,train_iter,loss,trainer):
    net.train()
    total_correct=0
    total_samples=0
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        correct, samples = accuracy(y_hat, y)
        total_correct += correct
        total_samples += samples
    return total_correct/total_samples
#训练模型
def train(net,train_iter,loss,trainer):
    for i in range(10):
        print("epoch %d,accuracy %f"%(i+1,train_epoch(net,train_iter,loss,trainer)))
if __name__ == '__main__':
    train(net,train_iter,loss,trainer)
