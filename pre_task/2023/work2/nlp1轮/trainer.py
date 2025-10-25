import torch
from torch import nn, optim
from torchtext import datasets
from torchtext import data
import numpy as np
import time
import random
import spacy
from torchtext import vocab
from IMDBmodel import RNN
from train import train
from test1 import eval

print('GPU:', torch.cuda.is_available())
seed = 123
torch.manual_seed(seed)

TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, vaild_data = train_data.split(random_state=random.seed(seed))

# print('len of train data:', len(train_data))
# print('len of test data:', len(test_data))

# word2vec, glove
TEXT.build_vocab(train_data, max_size=25_000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

# print(TEXT.vocab.freqs.most_common(20))

# print(TEXT.vocab.stoi)
print(LABEL.vocab.stoi)

batchsz = 50
device = torch.device('cuda')
train_iterator, vaild_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, vaild_data, test_data),
    batch_size=batchsz,
    device=device
)


rnn = RNN(len(TEXT.vocab), 100, 256)

pretrained_embedding = TEXT.vocab.vectors
print('pretrained_embedding:', pretrained_embedding.shape)
rnn.embedding.weight.data.copy_(pretrained_embedding)
print('embedding layer inited.')

optimizer = optim.Adam(rnn.parameters(), lr=0.001)
criteon = nn.BCEWithLogitsLoss().to(device)
rnn = rnn.to(device)


for epoch in range(4):
    # best_vaild_loss=float("inf")
    # print("Epoch {}/{}".format(epoch, 10))
    # print("train:")
    # train_acc,train_loss=train(rnn, train_iterator, optimizer, criteon)
    # print("vaild:")
    # vaild_acc,vaild_loss=eval(rnn, vaild_iterator, criteon)
    # if vaild_loss<best_vaild_loss:
    #     best_vaild_loss=vaild_loss
    #     torch.save(rnn.state_dict(),"bestIMDB.pt")
    #
    # print("test:")
    # test_acc,test_loss=eval(rnn, test_iterator, criteon)
    checkpoint = torch.load("bestIMDB.pt")
    rnn.load_state_dict(checkpoint)
    test_acc, test_loss = eval(rnn, test_iterator, criteon)