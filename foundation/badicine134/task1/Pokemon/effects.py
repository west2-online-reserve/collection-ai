from typing import TYPE_CHECKING, clear_overloads

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, duration: int | None) -> None:
        # 初始化效果持续时间
        self.duration = duration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        self.duration -= 1
        if self.duration > 0:
            print(f"{self.name} 效果持续时间减少，剩余: {self.duration} 回合")


class PoisonEffect(Effect):
    name = "剧毒"

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.hp -= self.damage
        if pokemon.hp < 0:
            pokemon.hp = 0
        print(f"{pokemon.name} 受到 {self.damage} 点中毒伤害！ 剩余 HP: {pokemon.hp}/{pokemon.max_hp}")

        if pokemon.hp == 0 and pokemon.alive:
            pokemon.alive = False
            print(f"{pokemon.name} 昏厥了！")


class HealEffect(Effect):
    name = "恢复"

    def __init__(self, amount: int, duration: None = None) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)

    def decrease_duration(self) -> None:
        pass

class  ParasiticHealEffect(Effect):
    name = "寄生种子恢复"

    def __init__(self, amount: int = 6, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        print("寄生种子吸取了对手的生命值！")
        pokemon.heal_self(self.amount)


class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.hp -= self.damage
        if pokemon.hp < 0:
            pokemon.hp = 0
        print(f"{pokemon.name} 正在承受烧伤！扣除 {self.damage} HP！ 剩余 HP: {pokemon.hp}/{pokemon.max_hp}")

        if pokemon.hp == 0 and pokemon.alive:
            pokemon.alive = False
            print(f"{pokemon.name} 昏厥了！")


class ShieldEffect(Effect):
    name = "护盾"

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pass


class ChargeEffect(Effect):
    name = "蓄力"

    def __init__(self, duration: None = None) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pass

    def decrease_duration(self) -> None:
        pass


class ParalysisEffect(Effect):
    name = "麻痹"

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pass