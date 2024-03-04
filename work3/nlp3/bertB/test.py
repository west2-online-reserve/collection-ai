import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

import time
from torch import optim
from bertB.src.modeling import BertForMaskedLM
from bertB.src.modeling import BertConfig
from bertB.src.utils import TextDataset,mask_tokens
from bertB.src.model2 import Model

device = torch.device('cuda')

path='/tmp/pycharm_project_392/test/ctrain.txt'
path1='/tmp/pycharm_project_392/test/ctest.txt'
text=[]
text1=[]
with open(path, 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        lin = line.strip()
        text.append(lin)

with open(path1, 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        lin = line.strip()
        text1.append(lin)



#加载预训练字典和分词方法
token = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='/tmp/pycharm_project_392/bert',
    cache_dir=None,
    force_download=False,
)
train_dataset=TextDataset(text)
test_dataset=TextDataset(text1)


def collate_fn(data):
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    #把第15个词固定替换为mask
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


bert=Model()
checkpoint = torch.load("/tmp/pycharm_project_392/bestBert22.pt")
bert.load_state_dict(checkpoint)
bert=bert.to(device)


#数据加载器

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=512,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)


def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


optimizer = optim.AdamW(bert.parameters(), lr=1e-5)
# criteon = nn.NLLLoss().to(device)
criteon = torch.nn.CrossEntropyLoss().to(device)



def eval(model, iterator, criteon):
    since = time.time()
    epoch_acc = 0
    epoch_loss = 0

    model.eval()
    with torch.no_grad():
     for i, (input_ids, attention_mask, token_type_ids,labels) in enumerate(tqdm(test_dataloader)):
        inputs=input_ids.to(device)
        labels=labels.to(device)

        token_type_ids=token_type_ids.to(device)
        attention_mask=attention_mask.to(device)

        print("*************************************")
        print(token.decode(input_ids[0]))
        print(input_ids[0])
        print(token.decode(labels[0]))
        pred=model(input_ids=inputs,attention_mask=attention_mask,token_type_ids= token_type_ids)

        loss = criteon(pred,labels)
        pred = pred.argmax(dim=1)
        print(token.decode(pred[0]))

        acc=binary_acc(pred,labels)

        time_elapsed = time.time() - since
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    acc1 = epoch_acc / len(iterator)
    loss1 = epoch_loss / len(iterator)
    print(
        "Time elapsed {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(f'acc:{acc1} ,loss:{loss1}')
    return acc1, loss1

for epoch in range(3):
    print("test：")
    test_acc, test_loss = eval(bert, test_dataloader,criteon)
    # print("test:")
    # test_acc,test_loss=eval(bert, test_iterator, criteon)
