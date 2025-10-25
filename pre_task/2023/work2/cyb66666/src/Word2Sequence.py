class Word2Sequence():
    UNK_TAG = 'UNK'  # 特殊字符
    PAD_TAG = 'PAD'  # 填充字符
    UNK = 0
    PAD = 1
    reverse_dict = {}

    def __init__(self):
        self.dict = \
            {
                self.UNK_TAG: self.UNK,
                self.PAD_TAG: self.PAD
            }  # 词典
        self.count_dict = \
            {

            }  # 计数词典

    def fit(self, sequence):  # 将句子中的单词保存到计数词典
        for word in sequence:
            self.count_dict[word] = self.count_dict.get(word, 0) + 1

    def build_vocab(self, min=None, max=None, max_features=None):
        if min is not None:
            self.count_dict = {word: count for word, count in self.count_dict.items() if min < count}
        if max is not None:
            self.count_dict = {word: count for word, count in self.count_dict.items() if max > count}
        sorted_list = sorted(self.count_dict.items(), key=lambda x: x[-1], reverse=True)
        if max_features is not None:
            sorted_list = sorted_list[:max_features]
        self.count_dict = dict(sorted_list)
        for word in self.count_dict:
            self.dict[word] = len(self.dict)
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def input_vocab(self,vocab_path):  # 直接从本地导入词典
        f = open(vocab_path)
        word_list = f.read().split('\n')
        number = len(word_list)
        for word in word_list:
            self.dict[word] = number + 2
            number = number - 1

    def transform(self, sequence, max_len=None):
        if max_len is not None:
            if len(sequence) < max_len:
                sequence = sequence + [self.PAD_TAG] * (max_len - len(sequence))
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sequence]

    def reverse_transform(self, nums):
        return [self.reverse_dict.get(num) for num in nums]

    def __len__(self):
        return len(self.dict)

    # w2s = Word2Sequence()
    # w2s.input_vocab()
    # #w2s.fit(['i','am','happy','a','day'])
    # #print(w2s.count_dict)
    # #print(w2s.dict)
    # print(w2s.transform(['i','am','happy','a','day']))

# w2s = Word2Sequence()
# w2s.input_vocab()
