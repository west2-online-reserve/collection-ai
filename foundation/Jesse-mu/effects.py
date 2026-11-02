from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect: 

    name: str = "Effect"

    def __init__(self, duration: int = 3):
        # 效果剩余持续回合数
        self.duration: int = duration

    def apply(self, pokemon: "Pokemon"):
        raise NotImplementedError

    def decrease_duration(self):
        """结算一次后减少持续时间。"""
        self.duration -= 1
        print(f"{self.name} effect duration decreased. Remaining: {self.duration}")


class PoisonEffect(Effect):
    """中毒：每回合损失固定生命值（无视防御）。"""

    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 3):
        super().__init__(duration)
        self.damage: int = damage

    def apply(self, pokemon: "Pokemon"):
        pokemon.hp -= self.damage
        print(
            f"{pokemon.name} takes {self.damage} poison damage! "
            f"Remaining HP: {pokemon.hp}/{pokemon.max_hp}"
        )
        if pokemon.hp <= 0:
            pokemon.alive = False
            print(f"{pokemon.name} has fainted!")


class BurnEffect(Effect):
    """烧伤：每回合受到额外伤害，持续若干回合。"""

    name = "Burn"

    def __init__(self, damage: int = 10, duration: int = 2):
        super().__init__(duration)
        self.damage: int = damage

    def apply(self, pokemon: "Pokemon"):
        pokemon.hp -= self.damage
        print(
            f"{pokemon.name} is burned and takes {self.damage} extra damage! "
            f"Remaining HP: {pokemon.hp}/{pokemon.max_hp}"
        )
        if pokemon.hp <= 0:
            pokemon.alive = False
            print(f"{pokemon.name} has fainted!")


class HealEffect(Effect):
    """回复：每回合恢复固定生命值。"""
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3):
        super().__init__(duration)
        self.amount: int = amount

    def apply(self, pokemon: "Pokemon"):
        pokemon.heal_self(self.amount)


class DrainEffect(Effect):
    """吸取：每回合吸取目标一定比例最大 HP 并回复自身（寄生种子）。

    效果挂在使用者身上，但每回合从注入的 target 处吸取生命。
    """

    name = "Drain"

    def __init__(self, percentage: float = 0.1, duration: int = 3):
        super().__init__(duration)
        self.percentage: float = percentage
        self.target: Optional["Pokemon"] = None

    def apply(self, pokemon: "Pokemon"):
        if self.target is None or not self.target.alive:
            return
        drained: int = int(self.target.max_hp * self.percentage)
        self.target.hp -= drained
        print(
            f"{self.target.name} loses {drained} HP to {pokemon.name}'s {self.name}! "
            f"Remaining HP: {self.target.hp}/{self.target.max_hp}"
        )
        if self.target.hp <= 0:
            self.target.alive = False
            print(f"{self.target.name} has fainted!")
        pokemon.heal_self(drained)


class ParalysisEffect(Effect):
    """麻痹：有概率使宝可梦本回合无法行动。"""

    name = "Paralysis"

    def __init__(self, skip_chance: int = 25, duration: int = 3):
        super().__init__(duration)
        self.skip_chance: int = skip_chance

    def apply(self, pokemon: "Pokemon"):
        if random.randint(1, 100) <= self.skip_chance:
            pokemon.paralysis_skip = True
            print(f"{pokemon.name} is paralyzed and may lose its turn!")
