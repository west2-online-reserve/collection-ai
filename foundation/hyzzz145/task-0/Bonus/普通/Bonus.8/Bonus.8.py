import random
def doudizhu():
    card= [str(i) for i in range(1,11)] + ['J','Q','K']
    card *= 4
    card += ['小王','大王']
    random.shuffle(card)
    dipai = card[:3]
    player1 = card[3:20]
    player2 = card[20:37]
    player3 = card[37:54]

    player1str  = ' '.join(player1)
    player2str = ' '.join(player2)
    player3str = ' '.join(player3)
    dipaistr = ' '.join(dipai)


    with open("player1.txt",'w',encoding='UTF-8') as f1:
        f1.write(player1str)
    with open("player2.txt",'w',encoding='UTF-8') as f2:
        f2.write(player2str)
    with open("player3.txt",'w',encoding='UTF-8') as f3:
        f3.write(player3str)
    with open("dipai.txt",'w',encoding='UTF-8') as f4:
        f4.write(dipaistr)         

doudizhu()
print("Finish")