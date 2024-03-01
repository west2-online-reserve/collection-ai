import torch
import module
import torchvision

train_set = torchvision.datasets.MNIST('.', transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = torchvision.datasets.MNIST('.', False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

my_module = module.my_module()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(my_module.parameters(), 0.01)

for i in range(1):
    for img, label in train_loader:
        optim.zero_grad()
        data = my_module(img)
        loss = loss_func(data, label)
        loss.backward()
        optim.step()
    total_correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            data = my_module(img)
            total_correct += (data.argmax(1) == label).sum().item()
    print(total_correct / len(test_set))  # accuracy:0.9097
torch.save(my_module, '.\module.pth')
