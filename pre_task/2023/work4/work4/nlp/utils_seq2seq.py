# coding=utf-8


from random import randint, shuffle, choice
from random import random as rand
import math
import numpy as np
import torch
import torch.utils.data
import json
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
pretrained_model_name_or_path='/tmp/pycharm_project_970/bert-base-chinese',
cache_dir=None,
force_download=False,
)



class TextDataset(Dataset):
    def __init__(self, text_samples):
        self.text_samples = text_samples

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        return self.text_samples[idx]


def mask_tokens(config, inputs, tokenizer):
    # 克隆输入用作标签
    labels = inputs.clone()

    # 构建概率矩阵，初始化为掩码的全局概率
    probability_matrix = torch.full(labels.shape, config.mlm_probability)

    # 获取特殊令牌的掩码
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    # 将特殊令牌的概率设置为0
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 识别每个序列的有效长度并计算中点，仅对后半部分应用掩码
    seq_lengths = inputs.ne(tokenizer.pad_token_id).sum(dim=1)
    half_seq_length = (seq_lengths / 2).ceil().long()  # 向上取整以确定中点
    for i in range(inputs.size(0)):
        # 只在序列的后半部分设置掩码概率
        probability_matrix[i, :half_seq_length[i]] = 0.0

    # 根据更新后的概率矩阵生成掩码位置
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # 对未掩码的位置设置-1

    # 替换掩码位置上的一部分为 [MASK] 标记
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, config.prob_replace_mask)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 选择性地将一些掩码位置替换为随机词汇
    current_prob = config.prob_replace_rand / (1 - config.prob_replace_mask)
    indices_random = torch.bernoulli(
        torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels, masked_indices
def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.vocab_words = None
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def init_skipgram_size_geo_list(self, p):
        # 初始化Skipgram大小的几何列表
        # 这个方法根据指定的概率p来初始化一个几何分布列表，该列表用于后续在生成mask位置时决定每个位置使用的n-gram大小
        if p > 0:
            g_list = []
            t = p
            for _ in range(self.skipgram_size):
                g_list.append(t)
                t *= (1-p)
            s = sum(g_list)
            self.skipgram_size_geo_list = [x/s for x in g_list]

    def __call__(self, instance):
        raise NotImplementedError

    # pre_whole_word: tokenize to words before masking
    # post whole word (--mask_whole_word): expand to words after masking
    # 获取掩码位置
    # 此方法的核心功能是确定哪些token位置将被掩码。它首先根据是否进行整词分割来创建一个范围列表。然后，它将特殊token（如'[CLS]'或'[SEP]'）排除在候选位置外，
    # 并可能根据不同段落来选择掩码位置。最后，根据n-gram大小决策和其他条件来选择最终的掩码位置
    def get_masked_pos(self, tokens, n_pred, add_skipgram=False, mask_segment=None, protect_range=None):
        if self.pre_whole_word:
            pre_word_split = _get_word_split_index(tokens, 0, len(tokens))
        else:
            # 每个token独立视为一个边界
            pre_word_split = list(range(0, len(tokens)+1))

        # 创建一个由每个词的开始和结束索引组成的元组列表
        span_list = list(zip(pre_word_split[:-1], pre_word_split[1:]))

        # candidate positions of masked tokens

        # 构建候选掩码位置列表
        cand_pos = []
        special_pos = set()
        if mask_segment:
            for i, sp in enumerate(span_list):
                sp_st, sp_end = sp
                # 如果遍历到最后一个词，就结束
                if (sp_end-sp_st == 1) and tokens[sp_st].endswith('SEP]'):
                    segment_index = i
                    break
        # 标记[CLS],[SEP]，后面就不会被mask掉
        for i, sp in enumerate(span_list):
            sp_st, sp_end = sp
            if (sp_end-sp_st == 1) and (tokens[sp_st].endswith('CLS]') or tokens[sp_st].endswith('SEP]')):
                special_pos.add(i)
            else:
                if mask_segment:
                    if ((i < segment_index) and ('a' in mask_segment)) or ((i > segment_index) and ('b' in mask_segment)):
                        cand_pos.append(i)
                else:
                    cand_pos.append(i)
        shuffle(cand_pos)

        #
        masked_pos = set()
        for i_span in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            cand_st, cand_end = span_list[i_span]
            if len(masked_pos)+cand_end-cand_st > n_pred:
                continue
            if any(p in masked_pos for p in range(cand_st, cand_end)):
                continue

            n_span = 1
            rand_skipgram_size = 0

            # ngram
            # if self.skipgram_size_geo_list:
            #     # sampling ngram size from geometric distribution
            #     rand_skipgram_size = np.random.choice(
            #         len(self.skipgram_size_geo_list), 1, p=self.skipgram_size_geo_list)[0] + 1
            # else:
            #     if add_skipgram and (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
            #         rand_skipgram_size = min(
            #             randint(2, self.skipgram_size), len(span_list)-i_span)

            for n in range(2, rand_skipgram_size+1):
                tail_st, tail_end = span_list[i_span+n-1]
                if (tail_end-tail_st == 1) and (tail_st in special_pos):
                    break
                if len(masked_pos)+tail_end-cand_st > n_pred:
                    break
                n_span = n
            st_span, end_span = i_span, i_span + n_span

            # if self.mask_whole_word:
            #     # pre_whole_word==False: position index of span_list is the same as tokens
            #     st_span, end_span = _expand_whole_word(
            #         tokens, st_span, end_span)

            skip_pos = None

            for sp in range(st_span, end_span):
                for mp in range(span_list[sp][0], span_list[sp][1]):
                    if not(skip_pos and (mp in skip_pos)) and (mp not in special_pos) and not(protect_range and (protect_range[0] <= mp < protect_range[1])):
                        masked_pos.add(mp)

        if len(masked_pos) < n_pred:
            shuffle(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos not in masked_pos:
                    masked_pos.add(pos)
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            # shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        return masked_pos

    # 替换掩码Token
    # 此方法将选定的token位置替换为掩码或其他随机token。如果启用了span_same_mask，连续掩码位置将使用相同的替换策略。
    def replace_masked_tokens(self, tokens, masked_pos):
        if self.span_same_mask:
            masked_pos = sorted(list(masked_pos))
        prev_pos, prev_rand = None, None
        for pos in masked_pos:
            if self.span_same_mask and (pos-1 == prev_pos):
                t_rand = prev_rand
            else:
                t_rand = rand()
            if t_rand < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif t_rand < 0.9:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
            prev_pos, prev_rand = pos, t_rand


# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


def truncate_tokens_signle(tokens_a, max_len):
    if len(tokens_a) > max_len-2:
        tokens_a = tokens_a[:max_len-2]
    return tokens_a


from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, filename, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行尾的换行符
                line = line.strip()
                # 计算中点位置
                mid_index = len(line) // 2
                # 将行分成大致相等的两部分
                part1 = line[:mid_index]
                part1=tokenizer.tokenize(part1.strip())
                part2 = line[mid_index:]
                part2 = tokenizer.tokenize(part2.strip())
                # 将分割后的两部分作为一个元组添加到列表中
                self.ex_list.append((part1, part2))


    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, tokens_b = instance[:2]
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)
        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(
            tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4Seq2seqDecode(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.max_tgt_length = max_tgt_length

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        segment_ids = [4]*(len(padded_tokens_a)) + [5]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask)
