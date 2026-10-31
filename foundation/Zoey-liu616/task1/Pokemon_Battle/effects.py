from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    def __init__(self, name: str, duration: int) -> None:
        self.name = name
        self.duration = duration

    def apply(self, pokemon: Pokemon) -> None:
        raise NotImplementedError

    def decrease_duration(self) -> None:
        self.duration -= 1


class HealEffect(Effect):
    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__("Heal", duration)
        self.amount = amount

    def apply(self, pokemon: Pokemon) -> None:
        healed = pokemon.heal_self(self.amount)
        if healed > 0:
            print(f"{pokemon.name}恢复了{healed}点HP！")


class PoisonEffect(Effect):
    def __init__(self, damage: int | None = None, duration: int = 3) -> None:
        super().__init__("Poison", duration)
        self.damage = damage

    def apply(self, pokemon: Pokemon) -> None:
        damage = self.damage
        if damage is None:
            damage = max(1, int(pokemon.max_hp * 0.1))
        actual_damage = pokemon.receive_status_damage(damage)
        if actual_damage > 0:
            print(f"{pokemon.name}受到了{actual_damage}点中毒伤害！")


class BurnEffect(Effect):
    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__("Burn", duration)
        self.damage = damage

    def apply(self, pokemon: Pokemon) -> None:
        actual_damage = pokemon.receive_status_damage(self.damage)
        if actual_damage > 0:
            print(f"{pokemon.name}受到了{actual_damage}点烧伤伤害！")


class ParasiticSeedEffect(Effect):
    def __init__(self, source: Pokemon, amount: int, duration: int = 3) -> None:
        super().__init__("Parasitic Seeds", duration)
        self.source = source
        self.amount = amount

    def apply(self, pokemon: Pokemon) -> None:
        actual_damage = pokemon.receive_status_damage(self.amount)
        if actual_damage <= 0:
            return

        print(f"寄生种子从{pokemon.name}吸取了{actual_damage}点HP！")
        if self.source.alive:
            healed = self.source.heal_self(actual_damage)
            if healed > 0:
                print(f"{self.source.name}恢复了{healed}点HP！")
