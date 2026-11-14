from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, duration: int) -> None:
        self.duration = duration

    def apply(self, pokemon: Pokemon) -> None:
        raise NotImplementedError

    def decrease_duration(self) -> None:
        self.duration -= 1


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: Pokemon) -> None:
        # 按最大 HP 的 10% 计算中毒伤害
        damage = max(1, int(pokemon.hp * 0.10))
        pokemon.hp = max(0, pokemon.hp - damage)
        if pokemon.hp == 0:
            pokemon.alive = False
        print(f"{pokemon.name} 受到中毒伤害 {damage} 点！剩余 HP：{pokemon.hp}")


class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: Pokemon) -> None:
        pokemon.hp = max(0, pokemon.hp - self.damage)
        if pokemon.hp == 0:
            pokemon.alive = False
        print(f"{pokemon.name} 受到烧伤伤害 {self.damage} 点！剩余 HP：{pokemon.hp}")


class ParalysisEffect(Effect):
    name = "麻痹"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: Pokemon) -> None:
        if random.uniform(0, 1) < 0.25:
            pokemon.skip_action_reason = "麻痹"


class HealEffect(Effect):
    name = "回复"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: Pokemon) -> None:
        if pokemon.hp < pokemon.max_hp:
            pokemon.heal_self(self.amount)


class ShieldEffect(Effect):
    name = "护盾"

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: Pokemon) -> None:
        pass