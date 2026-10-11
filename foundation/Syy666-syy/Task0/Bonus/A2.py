import random
order={'3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
    '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12,
    '2': 13, '小王': 14, '大王': 15}
cards=['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']*4
cards.append('小王')
cards.append('大王')
random.shuffle(cards)
player1=sorted(cards[0:17],key=lambda c:order[c],reverse=True )
player2=sorted(cards[17:34],key=lambda c:order[c],reverse=True )
player3=sorted(cards[34:51],key=lambda c:order[c],reverse=True )
others=sorted(cards[51:54],key=lambda c:order[c],reverse=True )
with open('player1.txt','w',encoding='utf-8')as f:
    f.write('玩家1:')
    f.write(' '.join(player1))
with open('player2.txt','w',encoding='utf-8')as f:
    f.write('玩家2:')
    f.write(' '.join(player1))
with open('player3.txt','w',encoding='utf-8')as f:
    f.write('玩家3:')
    f.write(' '.join(player1))
with open('others.txt','w',encoding='utf-8')as f:
    f.write('剩余的牌:')
    f.write(' '.join(others))
print('玩家1的牌',' '.join(player1))
print('玩家2的牌',' '.join(player2))
print('玩家3的牌',' '.join(player3))
print('剩余的牌',' '.join(others))

