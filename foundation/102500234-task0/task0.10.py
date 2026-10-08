import random

def get_poker():
    pokers = []
    hua_se = ["♠", "♥", "♣", "♦"]
    dian_shu = ["3","4","5","6","7","8","9","10","J","Q","K","A","2"]
    for h in hua_se:
        for d in dian_shu:
            pokers.append(h + d)
    pokers.append("小王")
    pokers.append("大王")
    return pokers

poker_weight = {
    "3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,
    "J":11,"Q":12,"K":13,"A":14,"2":15,"小王":16,"大王":17
}

def get_weight(card):
    if card in ("大王","小王"):
        dian = card
    else:
        dian = card[1:]
    return poker_weight[dian]

all_cards = get_poker()
random.shuffle(all_cards)

p1 = all_cards[0:17]
p2 = all_cards[17:34]
p3 = all_cards[34:51]
others = all_cards[51:]

p1_sorted = sorted(p1, key=get_weight, reverse=True)
p2_sorted = sorted(p2, key=get_weight, reverse=True)
p3_sorted = sorted(p3, key=get_weight, reverse=True)
others_sorted = sorted(others, key=get_weight, reverse=True)

with open("player1.txt","w",encoding="utf-8") as f:
    f.write(" ".join(p1_sorted))
with open("player2.txt","w",encoding="utf-8") as f:
    f.write(" ".join(p2_sorted))
with open("player3.txt","w",encoding="utf-8") as f:
    f.write(" ".join(p3_sorted))
with open("others.txt","w",encoding="utf-8") as f:
    f.write(" ".join(others_sorted))

print("发牌完成")
