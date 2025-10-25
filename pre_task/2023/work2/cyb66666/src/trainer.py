import torch
import pickle
import torch.nn as nn
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer,  epochs, lambda_reg, test_imdb_dataloader, train_imdb_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.lambda_reg = lambda_reg  # 0.001
        self.test_dataloader = test_imdb_dataloader
        self.train_dataloader = train_imdb_dataloader
        self.loss_value = []
        self.acc_rate = []

    def train(self):
        BCE = nn.BCELoss()
        optimizer = self.optimizer  # torch.optim.AdamW(net.parameters(), lr=0.0009, weight_decay=0.0001)
        for epoch in range(self.epochs):  # 进行训练
            time1 = time.time()
            e_loss = 0
            for (x, y) in self.train_dataloader:
                self.model.zero_grad()
                x = torch.as_tensor(x).to(device)
                y = torch.as_tensor(y).to(device)
                output = self.model.forward(x)
                loss = BCE(output, y)
                e_loss = loss
                loss.backward()
                optimizer.step()
            acc = self.evaluate()
            time2 = time.time()
            print('Epoch', epoch + 1, ':','acc =',acc,'loss =',e_loss.item(), 'time_cost =', round(time2 - time1, 2), 's')
            self.loss_value.append(e_loss.item())
            self.acc_rate.append(acc)

    def evaluate(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for (x, y) in self.test_dataloader:
                x = torch.as_tensor(x).to(device)
                y = torch.as_tensor(y).to(device)
                outputs = self.model.forward(x)
                for i, output in enumerate(outputs):  # i为标号(所以需要enumerate())，output为数据,
                    if y[i][torch.argmax(output)] == 1:
                        correct += 1
                    total += 1
            return correct / total

    def draw_loss_acc(self):
        plt.plot(range(0, self.epochs), self.loss_value)
        plt.plot(range(0, self.epochs), self.acc_rate)
        plt.xlabel('epoch')
        plt.ylabel('loss/accuracy')
        plt.show()

    def save_model(self):
        pickle.dump(self.model, open('./modul.pkl', 'wb'))
