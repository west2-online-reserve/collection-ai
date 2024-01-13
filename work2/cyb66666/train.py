import torch
import pickle
# from src import trainer
# from src import model
# from src import utils
from src import trainer, model, utils, Word2Sequence

w2s = Word2Sequence.Word2Sequence()
w2s.input_vocab('./imdb.vocab')
learning_rate = 0.0009
weight_decay = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, optimizer, epochs, lambda_reg, test_imdb_dataloader, imdb_dataloader
model_ = model.imdb_model(w2s).to(device)
optimizer = torch.optim.AdamW(model_.parameters(), lr=learning_rate, weight_decay=weight_decay)
epochs = 5
lambda_reg = 0.001
# test_imdb_dataloader = utils.imdb_dataloader('./IMDB.json','test',w2s)
# train_imdb_dataloader = utils.imdb_dataloader('./IMDB.json','train',w2s)
# pickle.dump(train_imdb_dataloader, open('./train_data.pkl', 'wb'))
# pickle.dump(test_imdb_dataloader, open('./test_data.pkl', 'wb'))
test_imdb_dataloader = pickle.load(open('./test_data.pkl', 'rb'))
train_imdb_dataloader = pickle.load(open('./train_data.pkl', 'rb'))  # json文件太大无法push，拆分成两个pkl文件后删除了imdb.json
Train = trainer.Trainer(model_, optimizer, epochs, lambda_reg, test_imdb_dataloader, train_imdb_dataloader)
if __name__ == '__main__':
    Train.train()
    Train.draw_loss_acc()
    Train.save_model()
