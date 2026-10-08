##尝试用所学的知识写一个斗地主随机发牌程序，将每个人的发牌以及多的三张牌的结果分别输出到 player1.txt，player2.txt，player3.txt，others.txt 四个文件中，可以不要求牌的花色
import random
ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

deck = ranks * 4 + ["小王", "大王"]
num = 1
for i in range(51):
    n = random.choice(deck)
    deck.remove(n)
    if num == 1:
        with open ("player1.txt","a",encoding="utf-8") as f:
            f.write(n)
    elif num == 2:
        with open ("player2.txt","a",encoding="utf-8") as f:
            f.write(n)
    elif num == 3:
        with open ("player3.txt","a",encoding="utf-8") as f:
            f.write(n)
    if num != 3:
        num = num +1
    else:
        num = 1
with open ("others.txt","a",encoding="utf-8") as f:
    f.write(str(deck))

