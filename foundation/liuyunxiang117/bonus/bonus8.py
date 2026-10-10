import random

cards = []

for name in ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]:
    for i in range(4):
        cards.append(name)

cards.append("小王")
cards.append("大王")

random.shuffle(cards)

player1 = cards[:17]
player2 = cards[17:34]
player3 = cards[34:51]
others = cards[51:]

with open("player1.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(player1))

with open("player2.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(player2))

with open("player3.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(player3))

with open("others.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(others))

print("发牌完成！")
#源自chatGPT