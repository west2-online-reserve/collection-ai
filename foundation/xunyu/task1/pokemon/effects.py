from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    """效果基类"""
    name: str

    def __init__(self, duration) -> None:
        # 初始化效果持续回合数
        self.duration = duration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        self.duration -= 1
        print(f"{self.name} 效果剩余回合: {self.duration}")


class PoisonEffect(Effect):
    # 中毒：每回合损失10%最大HP
    name = "Poison"

    def __init__(self, duration = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        damage = max(int(pokemon.max_hp * 0.1), 1)
        pokemon.receive_damage(damage)
        print(f"{pokemon.name} 受到 {damage} 点中毒伤害！")


class BurnEffect(Effect):
    # 烧伤：每回合固定10点伤害
    name = "Burn"

    def __init__(self, damage = 10, duration = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} 受到 {self.damage} 点烧伤伤害！")


class HealEffect(Effect):
    # 回复：每回合回血
    name = "Heal"

    def __init__(self, amount, duration = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} 回复了 {self.amount} 点HP！")
