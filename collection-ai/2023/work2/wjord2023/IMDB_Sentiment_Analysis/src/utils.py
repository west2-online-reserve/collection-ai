import json
import re

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


def json_data_set_processing(json_data_path: str, seq: str, label: str) -> []:
    """
    将json数据集中的语句seq和标签label拆分，并将数据集分为训练集，测试集，和校验集
    :param json_data_path:
    :param seq:
    :param label:
    :return X_train, y_train, X_test, y_test, X_check, y_check:
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_check = []
    y_check = []

    # 导入json数据集
    with open(json_data_path, 'r') as f:
        data = json.load(f)

    train_data = data['train']
    test_and_check_data = data['test']

    # 分出训练数据
    for item in train_data:
        X_train.append(item[seq])
        y_train.append(int(item[label]))

    # 使用sklearn分出测试集和校验集
    X_test_and_check = []
    y_test_and_check = []

    for item in test_and_check_data:
        X_test_and_check.append(item[seq])
        y_test_and_check.append(int(item[label]))
    X_test, X_check, y_test, y_check = train_test_split(X_test_and_check, y_test_and_check, test_size=0.3, train_size=0.7, random_state=102302125)

    return X_train, y_train, X_test, y_test, X_check, y_check


def clean_data(data: str) -> str:
    """
    清洗数据将数据集的大写英文转为小写并只保留小写英文和!?和空格
    :param data:
    :return:
    """
    data = data.lower()
    data = re.sub(r"[^a-z\-!? ]", "", data)
    return data


def vocab_data_set_processing(vocab_data_path: str) -> []:
    """
    将vocab文件的vocab列表提取出来
    :param vocab_data_path:
    :return:
    """
    with open(vocab_data_path) as vocab:
        vocab = vocab.readlines()
        vocab = [word.strip() for word in vocab]
        vocab = ['<PAD>'] + vocab
    return vocab


def build_onehot_dictionary(vocab: []) -> {}:
    """
    构建onehot词典
    :param vocab:
    :return:
    """
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def data_truncation(max_len: int, encoded_X: []):
    """
    截断数据到max_len长度
    :param max_len:
    :param encoded_X:
    :return:
    """
    truncate_encoded_X = [seq[0:max_len] for seq in encoded_X]
    return truncate_encoded_X


def data_padding(X: []) -> []:
    return pad_sequence([torch.tensor(seq) for seq in X], batch_first=True)


def build_data_loader(X, y):
    # 将数据集转化为tensor，并进行device agnostic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = torch.tensor(y).to(device)
    # 创建数据集
    data = TensorDataset(X, y)
    # 创建dataloader
    loader = DataLoader(data, batch_size=64, shuffle=True)
    return loader
