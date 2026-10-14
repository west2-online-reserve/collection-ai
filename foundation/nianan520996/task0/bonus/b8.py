import random
card=['A',2,3,4,5,6,7,8,9,10,'J','Q','K']
cards=card*4
cards.extend(['大王', '小王'])
random.shuffle(cards)
player1=cards[:17]
player2=cards[17:34]
player3=cards[34:51]
other=cards[51:]
open('player1.txt','w').write(str(player1))
open('player2.txt','w').write(str(player2))
open('player3.txt','w').write(str(player3))
open('other.txt','w').write(str(other))