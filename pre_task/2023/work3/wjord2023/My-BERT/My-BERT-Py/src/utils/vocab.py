import pickle
from collections import Counter

import tqdm


class WordVocab:
    def __init__(self, texts, min_freq=1):
        print("正在构建Vocab词表")
        counter = Counter()  # Counter可以接受迭代器，进行计数
        for line in tqdm.tqdm(texts):
            # # 检查line是不是一个列表， 用来判断是不是连续的（主要应该是处理英文单词的吧，我这里就不修改了，以后好搬用）
            # if isinstance(line, list):
            #     words = line
            # else:
            #     # 要处理中文词汇要去除' '
            #     words = list(line.replace('\n', '').replace('\t','').replace(' ',''))
            # 处理中文的方法
            words = [char for char in line]

            # 提取字并计数
            for word in words:
                counter[word] += 1

        self.freq = counter
        counter = counter.copy()

        # itos是index_to_string的缩写
        self.itos = ['[PAD]', '<unk>', '[CLS]', '[SEP]', '[MASK]']
        for token in self.itos:
            del counter[token]

        # 通过词频进行排序， counter.items()返回元组对，lambda无名函数用于获取次数
        words_and_freq = sorted(counter.items(), key=lambda tup: tup[0])
        # 然后按字母顺序进行排序
        words_and_freq.sort(key=lambda tup: tup[1], reverse=True)

        # 去除过少出现的字符
        for word, freq in words_and_freq:
            if freq < min_freq:
                break
            self.itos.append(word)

        self.stoi = {token: i for i, token in enumerate(self.itos)}

        self.unk_index = self.stoi['<unk>']
        self.cls_index = self.stoi['[CLS]']
        self.sep_index = self.stoi['[SEP]']
        self.mask_index = self.stoi['[MASK]']
        self.pad_index = self.stoi['[PAD]']

    def __len__(self):
        return len(self.itos)

    def to_seq(self, sentence, seq_len=None, with_len=False):
        words = [char for char in sentence]

        # 如果word在stoi中，就返回word中对应的数值，否则返回<unk>
        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            # 如果不足长度就补0
            seq += [0 for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        # 将一个整数序列转换回词的序列，如果idx小于itos的长度,就返回itos[idx]，否则返回'<%d>'%d是idx的数值
        words = [self.itos[idx] if idx < len(self.itos) else '<%d>' % idx for idx in seq if not with_pad or idx != 0]
        return "".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)
