import random
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

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} takes {self.damage} poison damage!")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} heals {self.amount} HP!")


class BurnEffect(Effect):
    name = "Burn"

    def __init__(self, duration: int, burn_damage: int) -> None:
        super().__init__(duration)
        self.burn_damage = burn_damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.burn_damage)
        print(f"{pokemon.name} takes {self.burn_damage} burn damage!")



class ParalyzeEffect(Effect):
    name = "Paralyzed"
    def __init__(self, duration: int) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name} is paralyzed and might not be able to move!")
        # 50% 几率使宝可梦无法行动
        
        if random.random() < 0.5:
            print(f"{pokemon.name} is fully paralyzed and cannot move!")
            pokemon.is_paralyzed = True
        else:
            pokemon.is_paralyzed = False


class ShieldEffect(Effect):
    name = "Shield"
    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)
        self.damage_reduction = 0.5  # 减少 50% 伤害

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name} is shielded and will take reduced damage this turn!")
        # 在这回合中减少受到的伤害
        pokemon.is_shielded = True

    def wear_off(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name}'s shield effect has worn off.")
        pokemon.is_shielded = False