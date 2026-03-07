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


# 麻痹 中毒 烧伤 治疗 减伤
class Paralysis(Effect):
    # 麻痹
    name = "Paralysis"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        # 麻痹效果：下一回合无法攻击
        pokemon.paralyzed = True  # 返回 True 表示麻痹成功，无法攻击


class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(pokemon.max_hp * 0.1)
        print(f"{pokemon.name} takes {self.damage} poison damage!")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} heals {self.amount} HP!")


class Burn(Effect):
    name = "Burn"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} takes {self.damage} burn damage!")


class DefenseDown(Effect):
    name = "Defense Down"

    def __init__(self, amount: int, duration: int = 1) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.reduce += self.amount
        print(f"{pokemon.name}'s defense decreased by {self.amount}!")


class NerveDamage(Effect):
    name = "Nerve Damage"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.paralyzed = True
        print(f"{pokemon.name} is paralyzed due to nerve damage!")
