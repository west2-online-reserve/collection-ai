import torch
import torch.nn as nn
import torch.nn.functional as F

class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.max(1, keepdim=True)
        loss.backward()
        optimizer.step()
        if batch_index %3000 == 0:
            print("Train Epoch : {} \t Loss :{:.6f}".format(epoch, loss.item()))

def test_model(model, device, test_loader,epoch):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("Test —— Average loss :{:.4f}, Accuracy : {:.3f}\n".format(
                test_loss, 100.0 * correct / len(test_loader.dataset)))
        if 100.0 * correct / len(test_loader.dataset)>99:
                torch.save(model, "mnist_{}.pth".format(epoch))
                print("模型已保存")
        return 100.0 * correct / len(test_loader.dataset)