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

class ParalyzeEffect(Effect):
    name = "Paralyze"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, opponent: "Pokemon") -> None:
        print(f"{opponent.name}被麻痹了")

class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 1) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, opponent: "Pokemon") -> None:
        print(f"{opponent.name}又受到了{self.damage}点伤害")
        opponent.receive_damage(self.damage)

# Squirtle的护盾效果
class ShieldEffect(Effect):
    name = "Shield"

    def __init__(self, damage_reduction: float = 0.5, duration: int = 1) -> None:
        super().__init__(duration)
        self.damage_reduction = damage_reduction

    def apply(self, user: "Pokemon") -> None:
        print(f"{user.name}的护盾效果生效")

    def on_receive_damage(self, damage: int):
        actual_damage = int(damage * (1 - self.damage_reduction))
        return actual_damage

# Charmander的烧伤效果
class BurnEffect(Effect):
    name = "Burn"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name}受到了{self.damage}点伤害")

# Charmander的蓄力效果
class ChargeEffect(Effect):
    name = "Charge"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name}正在蓄力...")