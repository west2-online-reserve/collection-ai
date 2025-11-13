from base.skills import Skill
from effects import SpecialEffect, effects
from misc.tools import printWithDelay
import random


class Ember(Skill):
    name = "火花"

    def __init__(self, amount: float = 1, chance: int = 10) -> None:
        super().__init__()
        self.amount = amount
        self.chance = chance

    def execute(self, user, opponent):
        if opponent.dodged():
            return False
        damage = user.attack * self.amount
        damage *= user.type_effectiveness(opponent)
        if random.randint(1, 100) <= self.chance:
            printWithDelay(f"{user.name} 使 {opponent.name} 陷入烧伤")
            opponent.add_status_effect(effects.BurnEffect())
        opponent.receive_damage(damage, self.name)
        return True


class Flame_Charge(Skill):
    name = "蓄能爆炎-蓄力"

    def __init__(self, amount: float = 3, chance: int = 80) -> None:
        super().__init__()
        self.amount = amount
        self.chance = chance

    def execute(self, user, opponent):
        user.add_status_effect(SpecialEffect.Flame(opponent))
        return False


class Flame_Charge_fire(Skill):
    name = "蓄能爆炎-发射"

    def __init__(self, amount: float = 3, chance: int = 80) -> None:
        super().__init__()
        self.amount = amount
        self.chance = chance

    def execute(self, user, opponent):
        if opponent.dodged():
            return False
        damage = user.attack * self.amount
        damage *= user.type_effectiveness(opponent)
        if random.randint(1, 100) <= 80:
            printWithDelay(f"{user.name} 使 {opponent.name} 陷入烧伤")
            opponent.add_status_effect(effects.BurnEffect())
        opponent.receive_damage(damage, self.name)
        return True
