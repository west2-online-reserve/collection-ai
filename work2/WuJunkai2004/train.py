'''
@brief: This file is used to train the model of cifar100.
@date: 2024/01/24
@author: Wu Junkai
'''

import pickle
import os

import numpy as np
import torch as th
from torch import nn, optim


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file_dir = 'cifar-100-python'
train_file = "train"
test_file = "test"

# load data
data = []
label = []
file_path = os.path.join(file_dir, train_file)
data_dict = unpickle(file_path)
data.append(data_dict[b'data'])
label.append(data_dict[b'fine_labels'])

data = np.concatenate(data)
label = np.concatenate(label)


# define model
class Cifar100(nn.Module):
    def __init__(self):
        super(Cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()

        self.linear1 = nn.Linear(128 * 5 * 5, 120)
        #self.linear1 = nn.Linear(128 * 5 * 5, 1024)
        self.linear2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x) # 激活
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 5 * 5)
        x = self.linear1(x)
        x = self.relu(x)

        return x


# define loss function and optimizer
model = Cifar100()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = th.device("cuda")

# move model to GPU
model.to(device)


# train
epochs = 10
batch_size = 64

#start training
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(len(data) // batch_size):
        # get the inputs
        inputs = th.from_numpy(data[i * batch_size: (i + 1) * batch_size]).float()
        inputs = inputs.view(-1, 3, 32, 32)
        labels = th.from_numpy(label[i * batch_size: (i + 1) * batch_size]).long()

        # move inputs and labels to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


# save model
th.save(model.state_dict(), "cifar100.pth")



# check the accuracy

file_path = os.path.join(file_dir, test_file)
data_dict = unpickle(file_path)
data = data_dict[b'data']
label = data_dict[b'fine_labels']

with th.no_grad():
    correct = 0
    total = len(data)
    for i in range(len(data)):
        inputs = th.from_numpy(data[i]).float()
        inputs = inputs.view(-1, 3, 32, 32)
        labels = th.from_numpy(np.array(label[i])).long()

        # move inputs and labels to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = th.max(outputs.data.cpu(), 1)  # move predicted to CPU
        correct += (predicted == labels.cpu()).sum().item()  # move labels to CPU

accuracy = correct / total
print('Accuracy: {:.2f}%'.format(accuracy * 100))
