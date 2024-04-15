from src.model.my_chinese_bert import MyChineseBERT
from src.trainer.trainer import MyBERTTrainer
from torch.utils.data import DataLoader

from src.utils.dataset import MyChineseBertDataset
from src.utils.vocab import WordVocab

if __name__ == '__main__':
    # 读取数据集
    import os

    if os.path.exists('src/data/word_vocab.pkl'):
        vocab = WordVocab.load_vocab('src/data/word_vocab.pkl')
    else:
        with open('src/data/text_for_vocab.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
        vocab = WordVocab(texts)
        vocab.save_vocab('src/data/word_vocab.pkl')

    vocab_size = len(vocab)
    print(vocab.itos[:50])
    print(vocab_size)

    vocab = WordVocab.load_vocab('src/data/word_vocab.pkl')
    dataset = MyChineseBertDataset('src/data/output.csv', vocab, 128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MyChineseBERT(vocab_size=len(vocab), hidden=256)
    trainer = MyBERTTrainer(model, vocab, dataloader)

    for epoch in range(10):
        trainer.train(epoch, dataloader)
        if epoch % 5 == 0:
            trainer.plot_train_loss()
            trainer.save_model('save_model/my_chinese_bert_%d.pth' % epoch)