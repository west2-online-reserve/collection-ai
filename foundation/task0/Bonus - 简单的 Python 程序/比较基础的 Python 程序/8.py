# 尝试用所学的知识写一个斗地主随机发牌程序
# 将每个人的发牌以及多的三张牌的结果分别输出到 
# player1.txt，player2.txt，player3.txt，others.txt 四个文件中
# 可以不要求牌的花色

import random

ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
cars = ['Red_Joker','Black_Jocker']
for i in ranks:
    for j in range(0,4):
        cars.append(i)

random.shuffle(cars)
f1 = open("player1.txt","w")
f1.write(" ".join(cars[:17]))
f1.close()

f2 = open("player2.txt","w")
f2.write(" ".join(cars[17:34]))
f2.close()

f3 = open("player3.txt","w")
f3.write(" ".join(cars[34:51]))
f3.close()

f4 = open("others.txt","w")
f4.write(" ".join(cars[51:]))
f4.close()