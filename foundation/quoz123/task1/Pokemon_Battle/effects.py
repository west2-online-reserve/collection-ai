from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str = "状态"

    def __init__(self, duration: int) -> None:
        if duration <= 0:
            raise ValueError("状态持续时间必须大于 0")
        self.duration = duration

    def apply(self, pokemon: Pokemon) -> None:
        raise NotImplementedError("子类必须实现 apply()")

    def decrease_duration(self) -> None:
        self.duration -= 1


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: Pokemon) -> None:
        damage = max(1, int(pokemon.max_hp * 0.1))
        print(f"{pokemon.name} 受到了中毒伤害！")
        pokemon.receive_status_damage(damage)


class HealEffect(Effect):
    name = "治疗"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        if amount <= 0:
            raise ValueError("治疗量必须大于 0")
        self.amount = amount

    def apply(self, pokemon: Pokemon) -> None:
        print(f"{pokemon.name} 受到治疗效果的恢复！")
        pokemon.heal_self(self.amount)


class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: Pokemon) -> None:
        print(f"{pokemon.name} 受到了烧伤伤害！")
        pokemon.receive_status_damage(self.damage)


class ParasiticSeedEffect(Effect):
    name = "寄生种子"

    def __init__(
        self,
        source: Pokemon,
        ratio: float = 0.1,
        duration: int = 3,
    ) -> None:
        super().__init__(duration)
        self.source = source
        self.ratio = ratio

    def apply(self, pokemon: Pokemon) -> None:
        drain = max(1, int(pokemon.max_hp * self.ratio))
        actual = pokemon.receive_status_damage(drain)
        if actual <= 0:
            return
        print(f"寄生种子从 {pokemon.name} 身上吸取了 {actual} 点 HP！")
        if self.source.alive:
            self.source.heal_self(actual)
