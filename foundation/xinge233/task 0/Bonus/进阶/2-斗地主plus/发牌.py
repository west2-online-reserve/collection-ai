Hua_Se = ["红桃", "黑桃", "方片", "梅花"]
Pai_Hao = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
joker = ["大王", "小王"]
cards = {}
for m in Hua_Se:
    for i, n in enumerate(Pai_Hao):
        cards[f"{m}{n}"] = i
# cards=cards + joker
cards.update(
    {
        "小王": 13,
        "大王": 14,
    }
)
# print(cards)
import random

bin = list(cards)
random.shuffle(bin)
# print(bin)
player1 = sorted(bin[0:17], key=lambda x: cards[x])
player2 = sorted(bin[17:34], key=lambda x: cards[x])
player3 = sorted(bin[34:51], key=lambda x: cards[x])
other = sorted(bin[51:54], key=lambda x: cards[x])
# print(rule)
# print(player1)
from pathlib import Path

base = Path(__file__).parent
with open(base / "player1.txt", "w+", encoding="utf-8") as p1:
    p1.write(" ".join(player1))
with open(base / "player2.txt", "w+", encoding="utf-8") as p2:
    p2.write(" ".join(player2))
with open(base / "player3.txt", "w+", encoding="utf-8") as p3:
    p3.write(" ".join(player3))
with open(base / "other.txt", "w+", encoding="utf-8") as p4:
    p4.write(" ".join(other))
