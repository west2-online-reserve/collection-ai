import random

def build_cards():
    lists=["A","2","3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    cards=[]
    for i in lists:
        cards.extend([i]*4)
    cards.append("大王")
    cards.append("小王")
    return cards

def distribute():
    cards=build_cards()
    random.shuffle(cards)
    player1=cards[0:17]
    player2=cards[17:34]
    player3=cards[34:51]
    others=cards[51:54]
    return player1,player2,player3,others

def write_cards(document,name,card):
    with open(document,"w",encoding="utf-8") as d:
        d.write(f"{name}的牌共{len(card)}张: \n")
        d.write("、".join(card)+"/n")

if __name__ == "__main__":
    p1, p2, p3, ot = distribute()
    write_cards("player1.txt", "玩家1", p1)
    write_cards("player2.txt", "玩家2", p2)
    write_cards("player3.txt", "玩家3", p3)
    write_cards("others.txt", "底牌", ot)

