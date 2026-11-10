from __future__ import annotations

import random
from typing import TYPE_CHECKING

import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str = "技能"

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        raise NotImplementedError("子类必须实现 execute()")

    def __str__(self) -> str:
        return self.name


class DamageSkill(Skill):
    def __init__(self, power: float) -> None:
        self.power = power

    def calculate_damage(self, user: Pokemon, opponent: Pokemon) -> int:
        effectiveness = user.type_effectiveness(opponent)
        return int(user.attack * self.power * effectiveness)

    def hit(self, user: Pokemon, opponent: Pokemon) -> bool:
        if opponent.try_dodge(user):
            return False
        opponent.receive_damage(self.calculate_damage(user, opponent))
        return True

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        self.hit(user, opponent)


class Thunderbolt(DamageSkill):
    name = "十万伏特"

    def __init__(self) -> None:
        super().__init__(1.4)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if not self.hit(user, opponent):
            return
        if opponent.alive and random.random() < 0.1:
            opponent.paralyzed = True
            print(f"{opponent.name} 麻痹了，下一次行动将被跳过！")


class QuickAttack(DamageSkill):
    name = "电光一闪"

    def __init__(self) -> None:
        super().__init__(1.0)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if not self.hit(user, opponent):
            return
        if opponent.alive and random.random() < 0.1:
            print("电光一闪触发了第二次攻击！")
            if not opponent.try_dodge(user):
                opponent.receive_damage(self.calculate_damage(user, opponent))


class SeedBomb(DamageSkill):
    name = "种子炸弹"

    def __init__(self) -> None:
        super().__init__(1.3)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if not self.hit(user, opponent):
            return
        if opponent.alive and random.random() < 0.15:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} 中毒了！")


class ParasiticSeeds(Skill):
    name = "寄生种子"

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return
        opponent.add_status_effect(effects.ParasiticSeedEffect(user))
        print(f"{opponent.name} 被种下了寄生种子！")


class AquaJet(DamageSkill):
    name = "水枪"

    def __init__(self) -> None:
        super().__init__(1.4)


class Shield(Skill):
    name = "护盾"

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        user.shield_active = True
        print(f"{user.name} 用水流形成了护盾，下一次受到的伤害减半！")


class Ember(DamageSkill):
    name = "火花"

    def __init__(self) -> None:
        super().__init__(1.0)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if not self.hit(user, opponent):
            return
        if opponent.alive and random.random() < 0.1:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} 烧伤了！")


class FlameCharge(Skill):
    name = "蓄能爆炎"

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if user.charging_skill is not self:
            user.charging_skill = self
            print(f"{user.name} 开始蓄力，下一回合释放蓄能爆炎！")
            return

        user.charging_skill = None
        if opponent.try_dodge(user, extra_dodge=20):
            return

        damage = int(user.attack * 3.0 * user.type_effectiveness(opponent))
        opponent.receive_damage(damage)
        if opponent.alive and random.random() < 0.8:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} 烧伤了！")


class Bubble(DamageSkill):
    name = "泡沫"

    def __init__(self) -> None:
        super().__init__(1.0)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if not self.hit(user, opponent):
            return
        if opponent.alive and random.random() < 0.25:
            opponent.flinched = True
            print(f"{opponent.name} 畏缩了，下一次行动将被跳过！")
