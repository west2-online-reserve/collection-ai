import torch
import torch.optim as optim
from torchvision import datasets, transforms
from res import Digit, train_model, test_model
#定义超参数
BATCH_SIZE = 64 #每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 #训练数据集的轮次

pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
#下载，加载数据
from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)



model = Digit().to(DEVICE)
#优化器
optimizer = optim.Adam(model.parameters())


#初始正确率
print("Initial Accuracy :{:.3f}\n".format(test_model(model, DEVICE, test_loader, 0)))

for epoch in range(1, EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader, epoch)
    print("-------------------------------")