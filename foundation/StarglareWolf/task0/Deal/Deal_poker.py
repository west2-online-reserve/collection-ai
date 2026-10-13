import random
#poker:(3~10,J,Q,K,A,2) * 4 + king * 2 = 54
RANK = ['3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', '小王', '大王']

SUITS = ['♠', '♥', '♣', '♦']

def create_deck():
    deck = []
    for rank in RANK[:-2]:
        for suit in SUITS:
            deck.append(suit + rank)
    deck.append("小王")
    deck.append("大王")
    return deck

def deck_weight(card):
    if card == "小王":
        return 13
    if card == "大王":
        return 14

    rank = card[1:] if card[0] in SUITS else card
    return RANK.index(rank)

def deal_card():
    deck = create_deck()
    random.shuffle(deck)

    player1 = deck[0:17]
    player2 = deck[17:34]
    player3 = deck[34:51]
    others = deck[51:]

    return player1, player2, player3, others

def sort_card(cards):
    return sorted(cards, key = deck_weight, reverse = True)

def write_files(filename, cards):
    with open(filename, 'w', encoding = 'utf-8') as f:
        for card in sort_card(cards):
            f.write(f'{card}  ')

        f.close()


file1 = "player1.txt"
file2 = "player2.txt"
file3 = "player3.txt"
others = "others.txt"
p1, p2, p3, other = deal_card()
write_files(file1, p1)
write_files(file2, p2)
write_files(file3, p3)
write_files(others, other)
