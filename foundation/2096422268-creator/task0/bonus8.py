import random

cards=[]
for rank in ["3","4","5","6","7","8","9","10","J","Q","K","A","2"]:
    cards.extend([rank]*4)
cards.extend(["小王","大王"])

random.shuffle(cards)

player1=cards[0:17]
player2=cards[17:34]
player3=cards[34:51]
others=cards[51:54]

for filename, hand in [
    ("player1.txt",player1),
    ("player2.txt",player2),
    ("player3.txt",player3),
    ("others.txt",others),
]:
    with open(filename,"w",encoding="utf-8") as f:
        f.write(" ".join(hand))