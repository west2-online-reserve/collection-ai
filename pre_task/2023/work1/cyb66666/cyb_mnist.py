import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
class Net(torch.nn.Module):#神经网络主体
    def __init__(self):
        super().__init__()#调用父类的init方法
        self.li1 = torch.nn.Linear(28*28,64)
        self.li2 = torch.nn.Linear(64, 64)#输入层，28*28，64神经元数量
        self.li3 = torch.nn.Linear(64, 64)
        self.li4 = torch.nn.Linear(64, 10)#3个隐藏层
    def forward(self,x):#前向传播
        x = torch.nn.functional.relu(self.li1(x))
        x = torch.nn.functional.relu(self.li2(x))
        x = torch.nn.functional.relu(self.li3(x))
        x = torch.nn.functional.log_softmax(self.li4(x),dim=1)#对数sortmax归一化
        return x
def evaluate(test_data,net):#评估正确率
    correct=0
    total=0;
    with torch.no_grad():
        for (x,y) in test_data:
            outputs=net.forward(x.view(-1,28*28))
            for i,output in enumerate(outputs):#i为标号(所以需要enumerate())，output为数据,
                if torch.argmax(output)==y[i]:
                    correct+=1
                total+=1
    return correct/total
train_data=DataLoader(MNIST("", train=True, transform=transforms.ToTensor(), download=True),batch_size=15,shuffle=True)
test_data=DataLoader(MNIST("", train=False, transform=transforms.ToTensor(), download=True),batch_size=15,shuffle=True)
net=Net()
print(f"初始准确率：{evaluate(test_data,net)}")
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#优化器 parameters提供模型参数
for epoch in range(3):#进行训练
    for (x,y) in train_data:
        net.zero_grad()
        output=net.forward(x.view(-1,28*28))
        loss=torch.nn.functional.nll_loss(output,y)#对数损失函数
        loss.backward()
        optimizer.step()#参数更新
    print(f'第{epoch+1}次准确率为：{evaluate(test_data,net)}')
i=0;
for (x,y) in test_data:#测试
    if i==3:
        break
    predict=torch.argmax(net.forward(x[0].view(-1,28*28)))
    plt.figure(i)
    plt.imshow(x[0].view(28, 28))
    plt.title("prediction: " + str(int(predict)))
    i+=1
plt.show()