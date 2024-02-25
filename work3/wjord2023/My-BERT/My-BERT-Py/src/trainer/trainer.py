import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn


class MyBERTTrainer:
    def __init__(self, model, vocab, train_dataloader, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.vocab = vocab
        self.train_dataloader = train_dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        print("模型总参数量：", sum(p.numel() for p in model.parameters()))
        self.train_losses = []

    def train(self, epochs, dataloader):
        data_itr = tqdm.tqdm(enumerate(dataloader), desc="正在载入%d epoch的数据" % epochs, total=len(dataloader))

        loss = 0.0

        for i, data in data_itr:
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data['input'], data['segment'])

            next_loss = self.criterion(next_sent_output, data['is_next'])

            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data['output'])

            loss = next_loss + mask_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            self.train_losses.append(loss)

            data_itr.set_postfix({'Epoch': epochs, 'Loss': loss})

        print("Epoch %d, Loss: %.4f" % (epochs, loss))

        return self.train_losses

    def plot_train_loss(self):
        plt.plot(np.arange(len(self.train_losses)), self.train_losses, label='Train Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
