# 用所学的知识写一个斗地主随机发牌程序
# 将每个人的发牌以及多的三张牌的结果
# 分别按照从大到小的顺序输出到
# player1.txt，player2.txt，player3.txt，others.txt 四个文件中

import random

values = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'Red_Joker':15,'Black_Jocker':14}

ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
cars = ['Red_Joker','Black_Jocker']
for i in ranks:
    for j in range(0,4):
        cars.append(i)

random.shuffle(cars)
f1 = open("player1.txt","w")
p1 = sorted(cars[:17],key = lambda x:values[x])
f1.write(" ".join(p1))
f1.close()

f2 = open("player2.txt","w")
p2 = sorted(cars[17:34],key = lambda x:values[x])
f2.write(" ".join(p2))
f2.close()

f3 = open("player3.txt","w")
p3 = sorted(cars[34:51],key = lambda x:values[x])
f3.write(" ".join(p3))
f3.close()

f0 = open("others.txt","w")
ot = sorted(cars[51:54],key = lambda x:values[x])
f0.write(" ".join(ot))
f0.close()