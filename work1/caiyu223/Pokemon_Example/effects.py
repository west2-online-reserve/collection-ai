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
        print(f"{self.name} 效果持续时间减少。剩余: {self.duration}")


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name} 受到了 {self.damage} 点中毒伤害!")


class HealEffect(Effect):
    name = "治疗"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} 恢复了 {self.amount} 点生命值!")

class ParalysisEffect(Effect):
    name = '麻痹'

    def __init__(self, duration: int = 4) -> None:
        super().__init__(duration)
        

    def apply(self,pokemon: 'Pokemon'):
        if self.duration%2 == 0:
            print('对手处于麻痹状态')
            pokemon.hit_rate -= 30
        if self.duration%2 == 1:
            pokemon.hit_rate += 30

class BurnEffect(Effect):
    name = '灼烧'

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        print("灼烧")
        pokemon.receive_true_damege(10)