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


from preprocessors import *
from bart_model1 import BartForConditionalGeneration,BartConfig
from utils import collate_fn

batch_size=128
device = torch.device("cuda:0")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
tokenizer = BertTokenizer.from_pretrained(
pretrained_model_name_or_path='/tmp/pycharm_project_970/bert-base-chinese',
cache_dir=None,
force_download=False,
)
# file="/tmp/pycharm_project_970/xiaoshuo2.txt"
file="/tmp/pycharm_project_970/xiaoshuo3.txt"

# bi_uni_pipeline = [Preprocess4Seq2seq(max_pred=20, mask_prob=0.2, vocab_words=list(tokenizer.vocab.keys()), indexer=tokenizer.convert_tokens_to_ids, max_len=30, mask_source_words=False, skipgram_prb=0.0, skipgram_size=1, mask_whole_word=False)]
#
# train_dataset =Seq2SeqDataset(
#          file, batch_size, tokenizer, max_len=30, bi_uni_pipeline=bi_uni_pipeline)
# train_sampler = RandomSampler(train_dataset, replacement=False)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
#                                                     num_workers=0, collate_fn=batch_list_to_batch_tensors, pin_memory=False)


#
text = []
with open(file, 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        lin = line.strip()
        text.append(lin)
train_dataset=TextDataset(text)


#数据加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)


# unilm_config = UnilmConfig(vocab_size=tokenizer.vocab_size)
bart_config = BartConfig(decoder_layers=2)
model=BartForConditionalGeneration(bart_config)

checkpoint = torch.load("/tmp/pycharm_project_970/bestBert5.pt")
model.load_state_dict(checkpoint)
model.to(device)
print(model)


optimizer=optim.Adam(model.parameters(),lr=5e-5,eps=1e-8)

t_total = int(len(train_dataloader) * batch_size)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total)

def train(model, iterator, optimizer):
    since = time.time()
    epoch_acc = 0
    epoch_loss = 0

    model.train()
#input_ids, attention_mask, token_type_ids,labels,masked_indices
    for i, batch in enumerate(tqdm(train_dataloader)):
        batch = [t.to(device) if t is not None else None for t in batch]
        # input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
        # loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=lm_label_ids,masked_pos=masked_pos,
        #                    masked_weights=masked_weights)
        #
        # print(loss)
        # masked_lm_loss = loss_tuple

        input_ids, attention_mask, token_type_ids,labels,tgt_mask,decoder_padding_mask=batch
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(labels.shape)
        # print(masked_indices.shape)
        # print(input_ids.shape)
        # print(tokenizer.decode(input_ids[0]))
        # print(labels.shape)
        # print(tokenizer.decode(labels[0]))
        # print(decoder_padding_mask.dtype)
        # print(tgt_mask.dtype)
        # print(tgt_mask.shape)
        # print(decoder_padding_mask.shape)
        # print(tgt_mask[0])
        # print(decoder_padding_mask[0])
        #
        # pre=model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids, decoder_attention_mask=decoder_padding_mask,tgt_mask=tgt_mask)
        # masked_lm_loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=labels,masked_pos=masked_indices

        #                       )
        # labels = input_ids.clone()  # 使用输入ID作为标签
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        # print(loss)

        # loss=masked_lm_loss

        loss.backward()
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()
        time_elapsed = time.time() - since
        epoch_loss += loss.item()

    # acc1 = epoch_acc / len(iterator)
    loss1 = epoch_loss / len(iterator)
    print(
        "Time elapsed {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(f'loss:{loss1}')
    return  loss1



# for epoch in range(100):
#     best_vaild_loss=float("inf")
#     print("Epoch {}/{}".format(epoch, 100))
#     print("train:")
#     train_loss=train(model,train_dataloader,optimizer)
#     # print("vaild:")
#     # vaild_acc,vaild_loss=eval(bert, vaild_iterator, criteon)
#     if train_loss<best_vaild_loss:
#         best_vaild_loss=train_loss
#         torch.save(model.state_dict(),"bestBert5.pt")

text1="第一章 小施无敌了"
print(text1)
for epoch in range(10):
    if epoch==1:
       pre = model.sample_generate_encoder_decoder(text1)
       print(pre)
       text1=text1+pre
    else:
        pre = model.sample_generate_encoder_decoder(text1[-20:])
        print(pre)
        text1=text1+pre

print(text1)

# # for epoch in range(60):
# #     best_vaild_loss=float("inf")
# #     print("Epoch {}/{}".format(epoch, 30))
# #     print("train:")
# #     text1="如同流淌的河流一般的能量在灵魂的视觉之中正"
# #     pre=model.sample_generate_encoder_decoder(text1)
# #
# #     print(pre)














