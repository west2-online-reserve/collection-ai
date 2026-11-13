from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, effect_round: int) -> None:
        # 初始化效果持续时间
        self.effect_round = effect_round


    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_effect_round(self) -> None:
        # 减少效果持续时间
        self.effect_round -= 1

    def effect_disappear(self) -> bool:
        if self.effect_round <= 0:
            return True
        return False


class PoisonEffect(Effect):
    name = "「中毒」"

    def __init__(self, damage_rate: int = 10, effect_round: int = 3) -> None:
        super().__init__(effect_round)
        self.damage_rate = damage_rate / 100

    # 损失 10% 生命值
    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(pokemon.max_hp * self.damage_rate)
        print(f"{pokemon.name}:{self.name},损失 {self.damage_rate * 100}% 生命值")


class HealEffect(Effect):
    name = "「治疗」"

    def __init__(self, amount: int, effect_round: int = 3) -> None:
        super().__init__(effect_round)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)

class ParalysisEffect(Effect):
    name = "「麻痹」"

    def __init__(self, effect_round: int = 1) -> None:
        super().__init__(effect_round)
        self.effect_round = effect_round

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name}:{self.name}")
        pokemon.paralyzed = True


class BurnedEffect(Effect):
    name = "「烧伤」"

    def __init__(self, effect_round: int = 2) -> None:
        super().__init__(effect_round)
        self.effect_round = effect_round

    def apply(self, pokemon: "Pokemon") -> None:
        print(f"{pokemon.name}:{self.name},每回合多承受 10 点伤害!")
        pokemon.receive_damage(10)