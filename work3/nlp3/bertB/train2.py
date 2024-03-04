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

# bertConfig=BertConfig(vocab_size=token.vocab_size)
# bertlm=BertForMaskedLM(bertConfig)
# checkpoint = torch.load("/tmp/pycharm_project_392/bestBert17.pt")
# bertlm.load_state_dict(checkpoint)
# bertlm=bertlm.to(device)


bert=Model()
checkpoint = torch.load("/tmp/pycharm_project_392/bestBert22.pt")
bert.load_state_dict(checkpoint)
bert=bert.to(device)



# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#数据加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=512,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=512,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)


def binary_acc(preds, y):
    """
    get accuracy
    """

    # correct = torch.eq(preds, y).float()
    correct = torch.eq(preds, y).float()
    # next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
    acc = correct.sum() / len(correct)
    return acc


optimizer = optim.AdamW(bert.parameters(), lr=1e-5)
# criteon = nn.NLLLoss().to(device)
criteon = torch.nn.CrossEntropyLoss().to(device)

def train(model, iterator, optimizer, criteon):
    since = time.time()
    epoch_acc = 0
    epoch_loss = 0

    model.train()
    for i, (input_ids, attention_mask, token_type_ids,labels) in enumerate(tqdm(train_dataloader)):
        inputs=input_ids.to(device)
        labels=labels.to(device)

        # print(token.decode(input_ids[0]))
        # print(input_ids[0])
        #
        # print(i)
        # print(token.decode(labels[0]))
        # print(labels[0])

        token_type_ids=token_type_ids.to(device)
        attention_mask=attention_mask.to(device)
        # print(inputs.shape)
        # print(labels.shape)


        # [seq, b] => [b, 1] => [b]o
        # pred = model(input_ids=input_ids,labels=labels)
        pred=model(input_ids=inputs,attention_mask=attention_mask,token_type_ids= token_type_ids)
        # print(token.decode(pred))
        # pred = torch.argmax(pred[0])
        # print("pred：")

        # print(pred[0])
        # print(token.decode(pred[0]))




        # pred=pred.view(-1, token.vocab_size)
        #
        # labels=labels.view(-1)

        loss = criteon(pred,labels)
        pred = pred.argmax(dim=1)
        # print(token.decode(pred[0]))


        acc=binary_acc(pred,labels)

        # pred=pred[:,:,0]
        # pred = torch.argmax(pred,dim=1)
        # pred=pred.view(-1, token.vocab_size)

        # labels=labels.view(-1)
        # print("pred：")
        # pred=pred.argmax(dim=1)
        # print(pred[0])
        # print(token.decode(pred[0]))



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

# for epoch in range(3):
#     best_vaild_loss=float("inf")
#     print("Epoch {}/{}".format(epoch, 3))
#     print("train:")
#     train_acc,train_loss=train(bert, train_dataloader, optimizer, criteon)
#     # print("vaild:")
#     # vaild_acc,vaild_loss=eval(bert, vaild_iterator, criteon)
#     if train_loss<best_vaild_loss:
#         best_vaild_loss=train_loss
#         torch.save(bert.state_dict(),"bestBert22.pt")
for epoch in range(3):
    print("test：")
    test_acc, test_loss = eval(bert, test_dataloader,criteon)
    # print("test:")
    # test_acc,test_loss=eval(bert, test_iterator, criteon)
