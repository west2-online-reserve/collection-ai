import random as rd

value={'A':12,'2':13,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'10':8,'J':9,'Q':10,'K':11,'小王':14,'大王':15}

card=['A','J','Q','K']

for i in range(2,11):
    card.append(str(i))

dl=[]
dl.extend(card*4)
dl.append('大王')
dl.append('小王')

rd.shuffle(dl)

player1=dl[0:17]
player2=dl[17:34]
player3=dl[34:51]
other=dl[51:54]

with open('player1.txt','w',encoding='utf-8') as f:
    player1.sort(key=lambda x:value[x])
    for i in player1:
        f.write(i+'\n')

with open('player2.txt','w',encoding='utf-8') as f:
    player2.sort(key=lambda x:value[x])
    for i in player2:
        f.write(i+'\n')

with open('player3.txt','w',encoding='utf-8') as f:
    player3.sort(key=lambda x:value[x])
    for i in player3:
        f.write(i+'\n')

with open('other.txt','w',encoding='utf-8') as f:
    other.sort(key=lambda x:value[x])
    for i in other:
        f.write(i+'\n')