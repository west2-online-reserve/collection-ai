from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, duration: int) -> None:
        # 初始化效果持续时间
        self.duration = duration

    #raise NotImplementedError 强制子类实现这个抽象方法。
    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        self.duration -= 1
        print(f"{self.name} effect duration decreased. Remaining: {self.duration}")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} heals {self.amount} HP!")

class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} takes {self.damage} poison damage!")

class ProtectEffect(Effect):
    name = "Protect"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage*0.5

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} takes {self.damage}  damage!")


class ParalysisEffect(Effect):
    name = "Paralysis"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        # 麻痹效果有50%几率无法行动
        import random
        if random.random() < 0.5:
            print(f"{pokemon.name} is paralyzed and cannot move!")
        else:
            print(f"{pokemon.name} is paralyzed but still manages to move!")


