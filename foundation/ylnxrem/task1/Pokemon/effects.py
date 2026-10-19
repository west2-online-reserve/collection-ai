from __future__ import annotations
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str
    type: str | None

    def __init__(self) -> None:
        self.duration: int = random.randint(1, 3)

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        raise NotImplementedError

    def decrease_duration(self) -> None:
        self.duration -= 1
        print(f"{self.name}效果持续时间减少。剩余持续回合: {self.duration}")


class PoisonEffect(Effect):
    name: str = "中毒"
    type: str = "毒"

    def __init__(self, damage: int = 10) -> None:
        super().__init__()
        self.damage: int = damage

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        print(f"{user.name}中毒")
        user.receive_damage(fixed_damage=self.damage)


class HealEffect(Effect):
    name: str = "回复"
    type: str | None = None

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount: int = amount

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        user.heal_self(self.amount)


class ParasiticEffect(Effect):
    name: str = "寄生"
    type: str = "草"

    def __init__(self) -> None:
        super().__init__()
        self.duration: int = 4

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        print("寄生效果发动")
        if opponent is not None:
            user.receive_damage(fixed_damage=int(user.max_hp / 10))
            opponent.heal_self(int(user.max_hp / 10))


class ParalysisEffect(Effect):
    name: str = "麻痹"
    type: str = "电"

    def __init__(self) -> None:
        super().__init__()

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        if random.randint(1, 100) <= 40:
            print(f"{user.name}被麻痹了，无法出招")
            user.action = False


class BurnEffect(Effect):
    name: str = "灼伤"
    type: str = "火"

    def __init__(self, damage: int = 10) -> None:
        super().__init__()
        self.damage: int = damage

    def apply(self, user: "Pokemon", opponent: "Pokemon" | None = None) -> None:
        print(f"{user.name}被灼伤")
        user.receive_damage(fixed_damage=self.damage)
