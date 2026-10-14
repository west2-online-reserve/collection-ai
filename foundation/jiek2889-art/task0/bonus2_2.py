SUITS = ["♠", "♥", "♣", "♦"]
RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
JOKERS = ["小王", "大王"]
ORDER = {(suit + face): (i, len(SUITS) - j) for i, face in enumerate(RANKS) for j, suit in enumerate(SUITS)}  # reference from deepseek
ORDER["小王"] = (len(RANKS), 1)  # 王最大，排在 3~2 之后
ORDER["大王"] = (len(RANKS), 2)

import random
import sys
from pathlib import Path


def decks():
    deck = []
    for suit in SUITS:
        for face in RANKS:
            deck.append(suit + face)
    deck += JOKERS
    return deck


def deal():
    cards = decks()
    random.shuffle(cards)
    player1 = hand(cards[0:17])
    player2 = hand(cards[17:34])
    player3 = hand(cards[34:51])
    others = hand(cards[51:54])
    return player1, player2, player3, others


def hand(cards):
    """把一手牌按从大到小排序"""
    return sorted(cards, key=lambda card: ORDER[card], reverse=True)


player1, player2, player3, others = deal()
Path("player1.txt").write_text("\n".join(player1) + "\n", encoding="utf-8-sig")
Path("player2.txt").write_text("\n".join(player2) + "\n", encoding="utf-8-sig")
Path("player3.txt").write_text("\n".join(player3) + "\n", encoding="utf-8-sig")
Path("others.txt").write_text("\n".join(others) + "\n", encoding="utf-8-sig")
print("已写入 player1.txt、player2.txt、player3.txt、others.txt")
