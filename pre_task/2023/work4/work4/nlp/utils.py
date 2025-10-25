
# coding=utf-8

import os
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import optim
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch import optim
from transformers import BertTokenizer
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig
from transformers import  get_linear_schedule_with_warmup
import time
from utils_seq2seq import TextDataset,mask_tokens

tokenizer = BertTokenizer.from_pretrained(
pretrained_model_name_or_path='/tmp/pycharm_project_970/bert-base-chinese',
cache_dir=None,
force_download=False,
)

class Config:
    def __init__(self):

       """
       :param mlm_probability: 被mask的token总数
       :param special_token_mask: 特殊token
       :param prob_replace_mask: 被替换成[MASK]的token比率
       :param prob_replace_rand: 被随机替换成其他token比率
       :param prob_keep_ori: 保留原token的比率
       """
       self.mlm_probability = 0.15
       self.special_tokens_mask = None
       self.prob_replace_mask = 0.8
       self.prob_replace_rand = 0.1
       self.prob_keep_ori = 0.1
       self.batch_size = 16

config=Config()

def collate_fn(data):
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=200,
                                   return_tensors='pt',
                                   return_length=True)

#     #input_ids:编码之后的数字
#     #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']


    # inputs, labels,masked_indices = mask_tokens(config, input_ids,tokenizer)
    split_point = input_ids.size(1) // 2  # 确定分割点
    inputs = input_ids[:, :split_point]  # 分割输入部分
    labels = input_ids[:, split_point:]  # 分割标签部分

    def create_decoder_attention_mask(input_ids, pad_token_id=tokenizer.pad_token_id):
        pad_mask = (input_ids != pad_token_id).long()
        seq_len = input_ids.size(1)
        seq_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device,dtype=torch.float32))
        decoder_attention_mask = pad_mask.unsqueeze(1) * seq_mask.unsqueeze(0)
        return decoder_attention_mask.float()

    def create_decoder_padding_mask(input_ids, pad_token_id=tokenizer.pad_token_id):
        # 创建填充掩码，填充部分为True，非填充部分为False
        decoder_padding_mask = (input_ids == pad_token_id).bool()
        return decoder_padding_mask

    decoder_padding_mask = create_decoder_padding_mask(input_ids)
    # print(decoder_padding_mask.dtype)


    decoder_attention_mask = create_decoder_attention_mask(input_ids)
    tgt_mask=decoder_attention_mask

    # decoder_padding_mask = input_ids == 0  # 填充位置为True
    # # 生成tgt_mask
    # seq_len = input_ids.size(1)
    # # tgt_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32))


    return inputs, attention_mask, token_type_ids,labels,tgt_mask,decoder_padding_mask