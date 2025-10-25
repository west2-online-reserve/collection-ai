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
        print(f"{self.name}效果持续时间: {self.duration}")


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        if pokemon.type=="草":
            return
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name}受到了{self.damage}的中毒伤害")


class HealEffect(Effect):
    name = "回复"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name}回复了{self.amount}点血量")
class ShieldEffect(Effect):
    name = "护盾"
    def __init__(self, amount: int, duration: int = 1) -> None:
        super().__init__(duration)
        self.amount = amount
    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.add_shield(self.amount)
        print(f"{pokemon.name}获得了{self.amount}的护盾")
class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        if pokemon.type=="火":
            return
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name}受到了{self.damage}的灼烧伤害")