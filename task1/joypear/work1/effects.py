from typing import TYPE_CHECKING

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
    # def __init__(self, damage: int = 10, duration: int = 3)
    def __init__(self, damage: int , duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name} takes {self.damage} poison damage!")
        pokemon.receive_damage(self.damage)



class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        #print(f"{pokemon.name} heals {self.amount} HP!")

class VampireEffect(Effect):
    name = "Vampire"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.vampire_self(self.amount)
        #print(f"{pokemon.name} is vampired {self.amount} HP!")

class ProtectEffect(Effect):
    name ="Protect"

    def __init__(self, ability: float , duration: int = 1) -> None:
        super().__init__(duration)
        self.ability = ability

    def apply(self, pokemon: "Pokemon") -> float:
        print(f"{pokemon.name} is protected {self.ability * 100}% damage by {"Shield"}!")
        return self.ability


# class ParalysisEffect():
#     name = "Paralysis"
#
#     def __init__(self, ability: float , duration: int = 1) -> None:
#         super().__init__(duration)
#         self.ability = ability
#
#     def apply(self, pokemon: "Pokemon") -> None:
#         print(f"{pokemon.name} is protected {self.ability * 100}% damage by {"Shield"}!")



