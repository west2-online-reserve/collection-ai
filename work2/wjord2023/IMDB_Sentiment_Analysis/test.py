from IMDB_Sentiment_Analysis.src import utils
from IMDB_Sentiment_Analysis.src.model import IMDBModel
MODEL_SAVE_PATH = 'IMDB_model.pth'

import torch


def test(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式

    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():  # 在测试过程中不计算梯度
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_check, y_check = utils.json_data_set_processing(
        'D:\Pycharm\AiProject\IMDB_Sentiment_Analysis\data\IMDB.json')
    X_check = [utils.clean_data(string) for string in X_check]
    vocab = utils.vocab_data_set_processing(r'D:\Pycharm\AiProject\IMDB_Sentiment_Analysis\data\imdb.vocab')
    word2idx, idx2word = utils.build_onehot_dictionary(vocab)
    encoded_X_check = [[word2idx.get(word, 0) for word in seq.split(' ')] for seq in X_check]
    encoded_X_check = utils.data_truncation(300, encoded_X_check)
    encoded_X_check = utils.data_padding(encoded_X_check)
    check_loader = utils.build_data_loader(encoded_X_check, y_check)
    model = IMDBModel(len(vocab), 128, 128, 2, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load('IMDB_model.pth'))
    loss_fn = torch.nn.CrossEntropyLoss
    check_loss, check_accuracy = test(model, check_loader, loss_fn)
    print(check_loss, check_accuracy)
