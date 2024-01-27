import torch
from torch import nn
from torch import optim

from IMDB_Sentiment_Analysis.src.model import IMDBModel
from IMDB_Sentiment_Analysis.src.trainer import train
from src import utils

if __name__ == '__main__':
    # 数据处理
    # 数据集拆分
    X_train, y_train, X_test, y_test, X_check, y_check = utils.json_data_set_processing(
        r'D:\Pycharm\AiProject\IMDB_Sentiment_Analysis\data\IMDB.json', 'seq', 'label')
    # 数据清洗
    X_train = [utils.clean_data(string) for string in X_train]
    X_test = [utils.clean_data(string) for string in X_test]
    # 构建词典
    vocab = utils.vocab_data_set_processing(r'D:\Pycharm\AiProject\IMDB_Sentiment_Analysis\data\imdb.vocab')
    word2idx, idx2word = utils.build_onehot_dictionary(vocab)
    # one-hot编码
    encoded_X_train = [[word2idx.get(word, 0) for word in seq.split(' ')] for seq in X_train]
    encoded_X_test = [[word2idx.get(word, 0) for word in seq.split(' ')] for seq in X_test]
    # 数据截断
    encoded_X_train = utils.data_truncation(300, encoded_X_train)
    encoded_X_test = utils.data_truncation(300, encoded_X_test)
    # 数据padding
    encoded_X_train = utils.data_padding(encoded_X_train)
    encoded_X_test = utils.data_padding(encoded_X_test)
    # loader
    train_loader = utils.build_data_loader(encoded_X_train, y_train)
    test_loader = utils.build_data_loader(encoded_X_test, y_test)

    # 模型处理
    model = IMDBModel(len(vocab), 128, 128, 2, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_losses, train_accuracies, val_losses, val_accuracies = train(model, train_loader, test_loader, loss_fn,
                                                                       optimizer, epochs)
