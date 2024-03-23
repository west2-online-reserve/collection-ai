from sklearn.model_selection import train_test_split
from src.model import BertForLastTokenClassification
from src.dataset import BertGenerateDataset
from src.trainer import Trainer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    input_ids, label_ids = zip(*sorted_batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-1)
    return input_ids, label_ids


def main():
    dataset = BertGenerateDataset('data/data_test.json')
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = BertForLastTokenClassification()
    trainer = Trainer(model=model, train_dataloader=train_loader, test_dataloader=test_loader)

    for epoch in range(10):
        trainer.train(epoch)
        trainer.test(test_loader)
        trainer.plot_train_and_test_loss()
        trainer.save_model('save_model/bert_generate_%d.pth' % epoch)


main()