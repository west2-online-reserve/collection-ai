from base.skills import Skill
from effects import SpecialEffect


SLEEP_TIME = 1


class Aqua_Jet(Skill):
    name = "水枪"

    def __init__(self, amount: float = 1.4) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user, opponent):
        if opponent.dodged():
            return
        damage = self.amount * user.attack
        damage *= user.type_effectiveness(opponent)
        opponent.receive_damage(damage, self.name)


class Shield(Skill):
    name = "护盾"

    def __init__(self, amount: float = 0.5) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user, opponent):
        a = SpecialEffect.DamageReductionEffect(self.amount)
        a.apply(user)
        user.add_status_effect(a)
        a.duration -= 1
