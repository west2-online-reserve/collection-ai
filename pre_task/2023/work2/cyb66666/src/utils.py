import json
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

max_len = 200


def mycollate(item):
    return list(zip(*item))


def seq_to_tensor(sequence):
    # 假设这里使用了简单的单词级别的标记化和嵌入，你可能需要使用更复杂的方法
    # 例如使用torchtext或Hugging Face的transformers库进行处理
    tokenized_sequence = sequence.split()  # 简单的按空格拆分
    embedding_dim = 300  # 假设嵌入维度为300
    embedding = torch.nn.EmbeddingBag(len(tokenized_sequence), embedding_dim)
    tensor = embedding(torch.tensor([range(len(tokenized_sequence))], dtype=torch.long))

    return tensor


def tokenization(text):
    text = re.sub('<.*?>', ' ', text, re.S)
    text = re.sub('[^A-Za-z]+', ' ', text).strip().lower()
    return [i.strip() for i in text.split()]


class IMDBDataset(Dataset):
    def __init__(self, json_path, train, w2s):
        self.w2s = w2s
        with open(json_path, 'r') as file:
            imdb_data = json.load(file)

        self.data = imdb_data[train]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.w2s.transform(tokenization(self.data[index]['seq']), max_len=max_len)
        label = [0., 1.] if self.data[index]['label'] == '1' else [1., 0.]

        # Placeholder for converting sequence to tensor, you may need tokenization and padding
        # For simplicity, assuming you have a function seq_to_tensor
        return sequence, label

def imdb_dataloader(json_path, train, w2s):
    imdb_dataset = IMDBDataset(json_path=json_path, train=train, w2s=w2s)
    imdb_dataloader1 = DataLoader(dataset=imdb_dataset, batch_size=256, shuffle=True, drop_last=True,collate_fn=mycollate)
    return imdb_dataloader1
