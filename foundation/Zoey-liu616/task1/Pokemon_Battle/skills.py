from __future__ import annotations

import random
from typing import TYPE_CHECKING

import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        raise NotImplementedError


class DamageSkill(Skill):
    def __init__(self, name: str, damage: int) -> None:
        super().__init__(name)
        self.damage = damage

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        damage = int(self.damage * effectiveness)
        opponent.receive_damage(damage)


class SeedBomb(DamageSkill):
    def __init__(self, damage: int = 50) -> None:
        super().__init__("Seed Bomb", damage)

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        opponent.receive_damage(int(self.damage * effectiveness))
        if opponent.alive and random.random() < 0.15:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name}陷入了中毒状态！")


class Ember(Skill):
    def __init__(self) -> None:
        super().__init__("Ember")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        opponent.receive_damage(int(user.attack * effectiveness))
        if opponent.alive and random.random() < 0.1:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}陷入了烧伤状态！")


class FlameCharge(Skill):
    def __init__(self) -> None:
        super().__init__("Flame Charge")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if user.charging_skill is None:
            user.charging_skill = self
            print(f"{user.name}开始蓄力！")
            return

        try:
            if opponent.try_dodge(user, 20):
                return

            effectiveness = user.type_effectiveness(opponent)
            damage = int(user.attack * 3.0 * effectiveness)
            opponent.receive_damage(damage)

            if opponent.alive and random.random() < 0.8:
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.name}陷入了烧伤状态！")
        finally:
            user.charging_skill = None


class AquaJet(Skill):
    def __init__(self) -> None:
        super().__init__("Aqua Jet")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        damage = int(user.attack * 1.4 * effectiveness)
        opponent.receive_damage(damage)


class Shield(Skill):
    def __init__(self) -> None:
        super().__init__("Shield")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        user.shield_active = True
        print(f"{user.name}获得了护盾，下一次受到的伤害减少50%！")


class ParasiticSeeds(Skill):
    def __init__(self, amount: int | None = None) -> None:
        super().__init__("Parasitic Seeds")
        self.amount = amount

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        amount = self.amount
        if amount is None:
            amount = max(1, int(opponent.max_hp * 0.1))

        opponent.add_status_effect(
            effects.ParasiticSeedEffect(user, amount, duration=3)
        )
        print(f"{opponent.name}被寄生种子附着了！")


class Thunderbolt(Skill):
    def __init__(self) -> None:
        super().__init__("Thunderbolt")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        damage = int(user.attack * 1.4 * effectiveness)
        opponent.receive_damage(damage)

        if opponent.alive and random.random() < 0.1:
            opponent.paralyzed = True
            print(f"{opponent.name}陷入了麻痹状态！")


class QuickAttack(Skill):
    def __init__(self) -> None:
        super().__init__("Quick Attack")

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if opponent.try_dodge(user):
            return

        effectiveness = user.type_effectiveness(opponent)
        damage = int(user.attack * effectiveness)
        opponent.receive_damage(damage)

        if opponent.alive and random.random() < 0.1:
            print("电光一闪触发了第二次攻击！")
            if user.alive and not opponent.try_dodge(user):
                opponent.receive_damage(damage)
