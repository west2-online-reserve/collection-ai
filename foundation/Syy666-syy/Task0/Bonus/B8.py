import random
cards=['3','4','5','6','7','8','9','10','J','Q','K','A','2']*4
cards.append('大王')
cards.append('小王')
random.shuffle(cards)
player1=cards[0:17]
player2=cards[17:34]
player3=cards[34:51]
others=cards[51:54]
with open('player1.txt','w',encoding='utf-8')as f:
    f.write('玩家1的牌：')
    f.write(''.join(player1))
with open('player2.txt','w',encoding='utf-8')as f:
    f.write('玩家2的牌：')
    f.write(''.join(player2))
with open('player3.txt','w',encoding='utf-8')as f:
    f.write('玩家3的牌：')
    f.write(''.join(player3))
with open('others.txt','w',encoding='utf-8')as f:
    f.write('剩余的三张牌')
    f.write(''.join(others))
print("玩家1的牌：",player1)
print("玩家2的牌：",player2)
print("玩家3的牌：",player3)
print("剩余的牌：",others)