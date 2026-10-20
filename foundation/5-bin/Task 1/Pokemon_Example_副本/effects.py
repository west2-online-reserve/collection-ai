from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str
    trigger_phase: str = "turn_end"  # 用于处理麻痹状态

    def __init__(self, duration: int) -> None:
        # 初始化效果持续时间
        self.duration: int = duration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self, pokemon: "Pokemon | None" = None) -> None:
        # 减少效果持续时间
        self.duration -= 1
        owner = f"{pokemon.display_name}'s " if pokemon is not None else ""
        print(f"{owner}{self.name} effect duration decreased. Remaining: {self.duration}")


class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        poison_damage = int(pokemon.max_hp * 0.1)
        pokemon.lose_hp(poison_damage)
        print(f"{pokemon.display_name} takes {poison_damage} poison damage!")


class HealEffect(Effect):
    name = "Heal"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount: int = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.display_name} heals {self.amount} HP!")


# 新增寄生种子效果类
class LeechSeedEffect(Effect):
    name = "LeechSeed"

    def __init__(self, source: "Pokemon", duration: int = 3) -> None:
        super().__init__(duration)
        self.source: "Pokemon" = source

    def apply(self, pokemon: "Pokemon") -> None:
        initial_hp = pokemon.hp
        leechseed_damage = int(pokemon.max_hp * 0.1)
        pokemon.lose_hp(leechseed_damage)
        leechseed_heal = initial_hp - pokemon.hp
        self.source.heal_self(leechseed_heal)


class ShieldEffect(Effect):
    name = "Shield"
    damage_multiplier: float = 0.5  # 受到伤害减半

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pass


class BurnEffect(Effect):
    name = "Burn"

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.lose_hp(10)  # 每回合扣除10点生命值


class ParalysisEffect(Effect):
    name = "Paralysis"
    trigger_phase: str = "before_action"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        # 麻痹效果可能会让宝可梦无法行动，这里仅打印提示
        print(f"{pokemon.display_name} is paralyzed and may not be able to move!")
