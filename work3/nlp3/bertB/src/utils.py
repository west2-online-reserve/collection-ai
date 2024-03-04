import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

import time
from torch import optim
from bertB.src.modeling import BertForMaskedLM
from bertB.src.modeling import BertConfig

device = torch.device('cuda')



class TextDataset(Dataset):
    def __init__(self, text_samples):
        self.text_samples = text_samples

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        return self.text_samples[idx]


def mask_tokens(config, inputs,tokenizer):

    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, config.mlm_probability)# 构建一个概率矩阵 `probability_matrix`，其中每个元素都是MLM任务中将一个词掩码的概率

    special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # 将特殊令牌的位置在概率矩阵中对应的值设为0

    masked_indices = torch.bernoulli(probability_matrix).bool() # 根据概率矩阵生成一个布尔掩码，用于确定哪些位置将被掩码。
    labels[~masked_indices] = -1  # 将不需要计算损失的非掩码位置的标签设置为-1,在损失函数计算的时候会忽略这些值

    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, config.prob_replace_mask)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) # 在被掩码的位置上80%的概率要用`[MASK]`替换掩码位置。
    current_prob = config.prob_replace_rand / (1 - config.prob_replace_mask)
    indices_random = torch.bernoulli(
        torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long) # 在被掩码的位置上确定10%的概率要用随机词替换掩码位置
    inputs[indices_random] = random_words[indices_random] # 在被掩码的位置上将10%的掩码位置替换为随机词

    return inputs, labels, masked_indices
