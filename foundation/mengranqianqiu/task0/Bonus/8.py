#这个我不太会写，大部分都是ai生成的
import random
class Card:
    def __init__(self, face):
        self.face = face
    def __repr__(self):
        return self.face
    def __lt__(self, other):
        order = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                 'J', 'Q', 'K', 'A', '小王', '大王']
        return order.index(self.face) < order.index(other.face)
class Poker:
    def __init__(self):
        faces = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                 'J', 'Q', 'K', 'A']
        self.cards = [Card(face) for face in faces for _ in range(4)]
        self.cards += [Card('小王'), Card('大王')]
        self.current = 0
    def shuffle(self):
        self.current = 0
        random.shuffle(self.cards)
    def deal(self):
        card = self.cards[self.current]
        self.current += 1
        return card
    @property
    def has_next(self):
        return self.current < len(self.cards)
class Player:
    def __init__(self, name):
        self.name = name
        self.cards = []
    def get_one(self, card):
        self.cards.append(card)
    def arrange(self):
        self.cards.sort()
    def show(self):
        print(f'{self.name}: {" ".join(str(c) for c in self.cards)}')
def main():
    poker = Poker()
    poker.shuffle()
    player1 = Player('玩家1')
    player2 = Player('玩家2')
    player3 = Player('玩家3')
    others = Player('底牌')
    players = [player1, player2, player3]
    for _ in range(17):
        for player in players:
            player.get_one(poker.deal())
    while poker.has_next:
        others.get_one(poker.deal())
    for player in players + [others]:
        player.arrange()
        player.show()
if __name__ == '__main__':
    main()
