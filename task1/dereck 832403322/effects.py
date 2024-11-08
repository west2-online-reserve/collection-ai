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
        print(f"{self.name} 的技能效果减少一回合. 剩余: {self.duration}回合")


class PoisonEffect(Effect):
    # 继承父类添加负面效果
    name = "Poison"

    # 给负面效果添加伤害值
    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage
    #输出宝可梦受到的伤害,更新血量
    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} 因为中毒而受到了 {self.damage} 点伤害!")


class HealEffect(Effect):
    # 继承父类添加正面效果
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        # 引入治疗量
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        # 治疗方法
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} 恢复了 {self.amount} 点血!")

class BurnEffect(Effect):
    # 继承父类添加中性效果
    name = "Burn"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} 受到 {self.damage} 点火焰伤害!")


