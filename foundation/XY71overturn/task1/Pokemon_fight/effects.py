from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, duration: int) -> None:
        # 初始化效果持续时间
        self.duration = duration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        self.duration -= 1
        print(f"{self.name} effect duration decreased. Remaining: {self.duration}")


class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, damage_rate: float=0.1, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage_rate = damage_rate
        self.damage = 0  # 初始化伤害值为 0

    def apply(self, pokemon: "Pokemon") -> None:
        self.damage = int(pokemon.max_hp * self.damage_rate)
        actual_damage = pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} takes {actual_damage} poison damage!")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)

class BurnEffect(Effect):
    name="Burn"


    def __init__(self,damage: int=10, duration:int=2) ->None:
        super().__init__(duration)
        self.damage=damage

    def apply(self,pokemon:"Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        
    def decrease_duration(self) -> None:
        self.duration -=1
        print(f"{self.name} effect duration decreased. Remaining: {self.duration}")

class ParalysisEffect(Effect):
    name="Paralysis"


    def __init__(self, duration:int=1) ->None:
        super().__init__(duration)

    def apply(self,pokemon:"Pokemon") -> None:
        # 50% chance to skip turn
        if random.random() < 0.5:
            print(f"{pokemon.name} is paralyzed and cannot move!")
            pokemon.skip_turn = True
        else:
            print(f"{pokemon.name} is paralyzed but can still move!")
