import random

POINTS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
JOKERS = ['junior', 'senior']
IDX = {cards: i for i, cards in enumerate(POINTS + JOKERS)}


def create_card():
    cards = []
    for i in POINTS:
        cards.extend([i] * 4)
    cards.extend(JOKERS)
    return cards


def sort_card(cards):
    return sorted(cards, key=lambda i: IDX[i])


def write_card(filename, cards):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(' '.join(sort_card(cards)))


def main():
    cards = create_card()
    random.shuffle(cards)

    player1 = cards[0: 17]
    player2 = cards[17: 34]
    player3 = cards[34: 51]
    others = cards[51: 54]

    write_card('player1.txt', player1)
    write_card('player2.txt', player2)
    write_card('player3.txt', player3)
    write_card('others.txt', others)


if __name__ == '__main__':
    main()
