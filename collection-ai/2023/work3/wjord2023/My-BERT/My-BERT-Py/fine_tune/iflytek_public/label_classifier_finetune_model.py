import json

import torch
from torch import nn

from fine_tune.iflytek_public.iflytek_dataset import IflytekDataset
from src.model.my_chinese_bert import MyChineseBERT
from src.utils.vocab import WordVocab


class LabelClassifierFineTuneModel(nn.Module):
    def __init__(self, pretrain_model, label_num):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.linear = nn.Linear(256, label_num)

    def forward(self, x):
        x = self.pretrain_model(x)
        return self.linear(x[:, 0])

if __name__ == '__main__':
    vocab = WordVocab.load_vocab('src/data/word_vocab.pkl')
    with open('labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
        label_num = len(labels)

    pre_train_model = MyChineseBERT(vocab_size=len(vocab), hidden=256).to('cuda' if torch.cuda.is_available() else 'cpu')
    model = LabelClassifierFineTuneModel(pre_train_model, label_num).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = IflytekDataset('train.json', vocab, 128)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = IflytekDataset('dev.json', vocab, 128)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            input, segment_info, label = data['input'], data['segment'], data['label']
            output = model(input, segment_info)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_acc = 0
        total_element = 0
        with torch.no_grad():
            for data in test_loader:
                input, segment_info, label = data['input'], data['segment'], data['label']
                output = model(input, segment_info)
                total_loss = criterion(output, label)
                total_acc += (output.argmax(dim=-1) == label).sum().item()
                total_element += len(label)
        print('Epoch:', epoch, 'Test Loss:', total_loss, 'Test Acc:', total_acc / total_element)


    # 保存微调后的模型
    torch.save(model.state_dict(), 'fine_tuned_model.pth')


