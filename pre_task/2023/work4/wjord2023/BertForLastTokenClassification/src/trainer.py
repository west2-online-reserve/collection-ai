import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        print("模型总参数量：", sum(p.numel() for p in model.parameters()))
        self.train_losses = []
        self.test_losses = []

    def train(self, epochs):
        data_itr = tqdm.tqdm(
            self.train_dataloader,
            desc="正在载入%d epoch的数据" % epochs,
            total=len(self.train_dataloader),
        )
        for input_ids, ladel_ids in data_itr:
            output = self.model(input_ids)
            loss = self.criterion(output, ladel_ids)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            self.train_losses.append(loss)
            data_itr.set_postfix({"Epoch": epochs, "train_Loss": loss})

        print("Epoch %d, Loss: %.4f" % (epochs, loss))

        return self.train_losses

    def test(self, epochs):
        self.model.eval()
        data_itr = tqdm.tqdm(
            self.train_dataloader,
            desc="正在进行%d epoch的测试" % epochs,
            total=len(self.test_dataloader),
        )
        test_loss = 0.0
        with torch.no_grad():
            for input_ids, label_ids in data_itr:
                output = self.model(input_ids)
                loss = self.criterion(output, label_ids)
                self.test_losses.append(test_loss)
                data_itr.set_postfix({"Epoch": epochs, "test_Loss": loss})
        return self.test_losses

    def plot_train_and_test_loss(self):
        plt.plot(
            np.arange(len(self.train_losses)), self.train_losses, label="train_loss"
        )
        plt.plot(np.arange(len(self.test_losses)), self.test_losses, label="test_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
