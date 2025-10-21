import story
import game

mainmoney = 100
mainmoney: int
# 你拥有的钱。可以用来买礼物
character1affinity = 0
character2affinity = 0
character3affinity = 0


class save:
    def __init__(self):
        self.money = mainmoney
        self.character1affinity = character1affinity
        self.character2affinity = character2affinity
        self.character3affinity = character3affinity


save0 = save()
if __name__ == 'manage':
    save0 = save()
