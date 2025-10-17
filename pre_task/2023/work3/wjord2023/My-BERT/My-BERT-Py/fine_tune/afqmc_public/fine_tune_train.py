import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import json

from fine_tune.afqmc_public.classifier_finetune_model import MyFineTunedModel
from fine_tune.afqmc_public.fine_tune_dataset import MyFineTuneDataset
from src.model.my_chinese_bert import MyChineseBERT
from src.utils.vocab import WordVocab

vocab = WordVocab.load_vocab('/content/drive/MyDrive/data/word_vocab.pkl')
bert_model = MyChineseBERT(vocab_size=len(vocab), hidden=32, n_layers=6, attn_heads=8, dropout=0.1)
bert_model.load_state_dict(torch.load('/content/drive/MyDrive/Train-Data/save_model/model_1.pth'))
model = MyFineTunedModel(bert_model)

batch_size = 32
lr = 1e-3
num_epochs = 10

with open('/content/drive/MyDrive/data/afqmc_public/train.json', 'r', encoding='utf-8') as file:
  train_data = [json.loads(line) for line in file]

with open('/content/drive/MyDrive/data/afqmc_public/dev.json', 'r', encoding='utf-8') as file:
  test_data = [json.loads(line) for line in file]


train_dataset = MyFineTuneDataset(train_data, vocab, 128)
test_dataset = MyFineTuneDataset(test_data, vocab, 128)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for input_ids, segment_info, label_tensor in tqdm(train_loader):
        input_ids, segment_info, labels = input_ids.to(device), segment_info.to(device), label_tensor.to(device)
        optimizer.zero_grad()
        output = model(input_ids, segment_info)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch} Train Loss: {train_loss / len(train_loader)}')

    model.eval()
    test_loss = 0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for input_ids, attention_mask, segment_info, label_tensor in tqdm(test_loader):
            input_ids, attention_mask, segment_info, label_tensor = input_ids.to(device), attention_mask.to(device), segment_info.to(device), label_tensor.to(device)
            output = model(input_ids, attention_mask, segment_info)
            loss = criterion(output, label_tensor)
            test_loss += loss.item()
            total_correct += (output.argmax(1) == label_tensor).sum().item()
            total_count += len(label_tensor)
    print(f'Epoch {epoch} Test Loss: {test_loss / len(test_loader)}')
    print(f'Epoch {epoch} Test Accuracy: {total_correct / total_count}')

torch.save(model.state_dict(), '/content/drive/MyDrive/Train-Data/save_model/model_fine_tune_1.pth')


