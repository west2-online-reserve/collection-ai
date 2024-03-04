import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

import time
from torch import optim
from bertB.src.modeling import BertForMaskedLM
from bertB.src.modeling import BertConfig
from bertB.src.utils import TextDataset,mask_tokens

device = torch.device('cuda')

path = '/tmp/pycharm_project_392/tieba7.txt'
text = []
with open(path, 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        lin = line.strip()
        text.append(lin)

# 加载预训练字典和分词方法
token = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='/tmp/pycharm_project_392/bert',
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
bertConfig=BertConfig(vocab_size=token.vocab_size)
bertlm=BertForMaskedLM(bertConfig)
checkpoint = torch.load("/tmp/pycharm_project_392/bestBert17.pt")
bertlm.load_state_dict(checkpoint)
bertlm=bertlm.to(device)


def collate_fn(data):
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=100,
                                   return_tensors='pt',
                                   return_length=True)

#     #input_ids:编码之后的数字
#     #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    inputs, labels,masked_indices = mask_tokens(config, input_ids,token)


    return input_ids, attention_mask, token_type_ids,inputs,labels,masked_indices
#
train_dataset=TextDataset(text)
#数据加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=128,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds.argmax(dim=1), y).float()
    acc = correct.sum() / len(correct)
    return acc

optimizer = optim.AdamW(bertlm.parameters(), lr=1e-4, weight_decay=0)
# criteon = nn.NLLLoss().to(device)
criteon = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)


def train(model, iterator, optimizer, criteon):
    since = time.time()
    epoch_acc = 0
    epoch_loss = 0

    model.train()
    for i, (input_ids, attention_mask, token_type_ids,inputs,labels,masked_indices) in enumerate(tqdm(train_dataloader)):
        inputs=inputs.to(device)
        labels=labels.to(device)
        masked_indices=masked_indices.to(device)
        # print(masked_indices[0])



        input_ids=input_ids.to(device)
        # print(token.decode(input_ids[0]))
        # print(input_ids[0])

        # print(token.decode(labels[0]))
        # print(labels[0])

        token_type_ids=token_type_ids.to(device)
        attention_mask=attention_mask.to(device)
        # print(inputs.shape)
        # print(labels.shape)


        # [seq, b] => [b, 1] => [b]
        # pred = model(input_ids=input_ids,labels=labels)
        pred=model(input_ids=inputs,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # print(pred.shape)
        # print(token.decode(pred))
        # pred = torch.argmax(pred)
        # print(token.decode(pred))




        pred=pred.view(-1, token.vocab_size)

        labels=labels.view(-1)

        loss = criteon(pred,labels)
        # acc1=binary_acc(pred.view(-1, token.vocab_size),labels.view(-1))
        acc1=0

        # pred=pred[:,:,0]
        # pred = torch.argmax(pred,dim=1)
        # pred=pred.view(-1, token.vocab_size)

        # labels=labels.view(-1)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_elapsed = time.time() - since
        epoch_loss += loss.item()

    # acc1 = epoch_acc / len(iterator)
    loss1 = epoch_loss / len(iterator)
    print(
        "Time elapsed {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(f'acc:{acc1} ,loss:{loss1}')
    return acc1, loss1

for epoch in range(30):
    best_vaild_loss=float("inf")
    print("Epoch {}/{}".format(epoch, 30))
    print("train:")
    train_acc,train_loss=train(bertlm, train_dataloader, optimizer, criteon)
    # print("vaild:")
    # vaild_acc,vaild_loss=eval(bert, vaild_iterator, criteon)
    if train_loss<best_vaild_loss:
        best_vaild_loss=train_loss
        torch.save(bertlm.state_dict(),"bestBert17.pt")

    # print("test:")
    # test_acc,test_loss=eval(bert, test_iterator, criteon)

train(bertlm,train_dataloader,optimizer,criteon)
