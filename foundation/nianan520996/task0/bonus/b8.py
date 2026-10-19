import random
card=['A',2,3,4,5,6,7,8,9,10,'J','Q','K']
cards=card*4
cards.extend(['大王', '小王'])
random.shuffle(cards)
player1=cards[:17]
player2=cards[17:34]
player3=cards[34:51]
other=cards[51:]
with open('player1.txt','w')as f:
    f.write(str(player1))
with open('player2.txt','w')as f:
    f.write(str(player2))
with open('player3.txt','w')as f:
    f.write(str(player3))
with open('other.txt','w')as f:
    f.write(str(other))