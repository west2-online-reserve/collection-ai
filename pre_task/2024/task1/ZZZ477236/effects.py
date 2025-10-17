from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
import random

class Effect:
    name: str

    def __init__(self, duration: int) -> None:
        self.duration = duration

    def apply(self, pokemon: "Pokemon") -> None:
        raise NotImplementedError("Each effect must implement its own apply method.")

    def decrease_duration(self, pokemon: "Pokemon") -> None:
        # 通用持续时间减少方法
        self.duration -= 1
        print(f"{self.name}效果的持续时间减少。剩余: {self.duration}")
        if self.duration <= 0:
            self.end_effect(pokemon)

    def end_effect(self, pokemon: "Pokemon") -> None:
        # 通用效果移除逻辑
        print(f"{pokemon.name}的{self.name}效果已消失。")
        if hasattr(pokemon, "remove_status_effect"):
            pokemon.remove_status_effect(self)


class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.receive_damage(self.damage)
        print(f"{pokemon.name}受到{self.damage}点中毒伤害！")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name}恢复了{self.amount}点HP！")


class BurnEffect(Effect):
    name = "Burn"

    def __init__(self, burn_damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.burn_damage = burn_damage

    def apply(self, target: "Pokemon") -> None:
        target.receive_damage(self.burn_damage)
        print(f"{target.name}被灼伤，受到{self.burn_damage}点灼伤伤害！")


class ShieldEffect(Effect):
    name = "Shield"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, user: "Pokemon") -> None:
        user.shield_active = True
        print(f"{user.name}被护盾保护！")


class ParalysisEffect(Effect):
    name = "Paralysis"

    def __init__(self, duration: int = 3, paralysis_chance: int = 25) -> None:
        super().__init__(duration)
        self.paralysis_chance = paralysis_chance

    def apply(self, pokemon: "Pokemon") -> None:
        # 判断宝可梦是否因麻痹而无法行动
        if random.randint(1, 100) <= self.paralysis_chance:
            print(f"{pokemon.name}因麻痹而无法行动！")
            pokemon.can_act = False
        else:
            pokemon.can_act = True


class ConfusionEffect(Effect):
    name = "Confusion"

    def __init__(self, duration: int = 3, confusion_chance: int = 50) -> None:
        super().__init__(duration)
        self.confusion_chance = confusion_chance

    def apply(self, pokemon: "Pokemon") -> None:
        # 检查是否混乱自己攻击
        if random.randint(1, 100) <= self.confusion_chance:
            # 造成自伤，假设为自身攻击力的一部分
            self_damage = pokemon.attack // 2
            pokemon.receive_damage(self_damage)
            print(f"{pokemon.name}因混乱而攻击自己，受到{self_damage}点伤害！")
        else:
            print(f"{pokemon.name}克服了混乱状态，可以正常行动。")
