import random

import jieba
import torch
from torch.utils.data import Dataset


class MyChineseBertDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding='utf-8'):
        """
        创建数据集
        :param corpus_path: 数据集的路径，corpus的意思是语料库
        :param vocab: 词典
        :param seq_len: 指定的序列长度
        :param encoding: 编码方式，默认为utf-8
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.corpus_path = corpus_path
        self.encoding = encoding

        # 将数据集的每一行分割并存储到 self.lines中, 以'\t'分割，用于下面的__getitem__方法，以强化其在不同任务中的通用性
        with open(corpus_path, 'r', encoding=encoding) as f:
            self.lines = [line[:-1].split('\t') for line in f.readlines()]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # is_next_label: 1表示是下一句，0表示不是下一句
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # 在t1_random和t2_random的首尾分别加上[CLS]和[SEP]标志
        t1 = [self.vocab.cls_index] + t1_random + [self.vocab.sep_index]
        t2 = t2_random + [self.vocab.sep_index]

        # 在句子的标签中加上[PAD]标志, 使得句子的长度都为seq_len
        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # segment_label 是一个列表，用于指示句子中的每个词属于哪个句子, segment的意思是段落
        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]

        # 输入由两个句子组成, 将两个句子拼接在一起，并且截断到指定长度
        model_input = (t1 + t2)[:self.seq_len]
        model_label = (t1_label + t2_label)[:self.seq_len]

        # 如果输入的长度小于指定长度，则用[PAD]标签来填充
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(model_label))]
        model_input.extend(padding), model_label.extend(padding), segment_label.extend(padding)

        # 将模型输入和标签转换回原始句子
        # original_t1 = self.vocab.from_seq(t1, join=True, with_pad=True)
        # original_t2 = self.vocab.from_seq(t2, join=True, with_pad=True)

        # 返回模型的输入和标签
        output = {
            'input': torch.tensor(model_input),
            'output': torch.tensor(model_label),
            'segment': torch.tensor(segment_label),
            'is_next': torch.tensor(is_next_label)
        }

        return {key: value.clone().detach() for key, value in output.items()}

        # 生成随机遮蔽的句子，返回遮蔽后的句子和标签

    def random_word(self, sentence):
        # 对中文token进行分割
        tokens = [char for char in sentence]
        output_label = []

        for i, token in enumerate(tokens):
            # 将15%的词进行替换
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% 的概率进行替换为[MASK]
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% 的概率进行替换为随机字, 增加模型的鲁棒性
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% 的概率不进行替换，也是为了增加鲁棒性
                else:
                    # stoi是string to index的缩写，如果没有找到这个字返回'<unk>'
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                # 记录被替换的词, 用于计算loss
                output_label.append(self.vocab.stoi.get(token, 1))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    # 随机获取一对句子和一个标签
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # 50%的概率获得连续的句子，标记标签1，这样做是为了提升在预测下一句任务上的能力
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    # 获取数据集的一行
    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    # 随机获取数据集的一行
    def get_random_line(self):
        return self.lines[random.randrange(self.corpus_lines)][0]