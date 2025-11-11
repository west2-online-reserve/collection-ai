from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pokemon import Pokemon


class Buff(ABC):
    name: str
    default_duration: int

    def __init__(self, duration: int):
        self.duration: int = duration

    @abstractmethod
    def apply(self, pokemon: "Pokemon"):
        raise NotImplemented

    def decrease_duration(self):
        self.duration -= 1


class PoisonBuff(Buff):
    name: str = "中毒"

    default_duration = 114514

    def __init__(self):
        super().__init__(self.default_duration)

    def apply(self, pokemon: "Pokemon"):
        pokemon.health_point -= pokemon.max_health_point / 10


class ParasiticBuff(Buff):
    name: str = "寄生"

    default_duration = 3

    def __init__(self):
        super().__init__(self.default_duration)

    def apply(self, pokemon: "Pokemon"):
        suck_blood = int(pokemon.enemy.max_health_point / 10)
        pokemon.enemy.health_point -= suck_blood
        pokemon.health_point += suck_blood
        print(
            f"[{pokemon.name}]使用{self.name}从[{pokemon.enemy.name}]吸收了{suck_blood}点生命值！(剩余HP: {pokemon.health_point})"
        )


class ParalysisBuff(Buff):
    name: str = "麻痹"

    default_duration = 1

    def __init__(self):
        super().__init__(self.default_duration)

    def apply(self, pokemon: "Pokemon"):
        print(f"[{pokemon.name}]因{self.name}跳过这一回合！")
        pokemon.skip_turn = True


class ShieldBuff(Buff):
    name: str = "护盾"

    default_duration = 1

    def __init__(self):
        super().__init__(self.default_duration)

    def apply(self, pokemon: "Pokemon"):
        pokemon.damage_reduction = 50


class BurnBuff(Buff):
    name: str = "烧伤"

    default_duration = 2

    def __init__(self):
        super().__init__(self.default_duration)

    def apply(self, pokemon: "Pokemon"):
        damage = 10
        pokemon.health_point -= damage
        print(f"[{pokemon.name}]因{self.name}减少了{damage}点生命值！")
