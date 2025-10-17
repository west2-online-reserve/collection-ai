import random

from enums import Type
from models import Pokemon, Skill, ShieldBuff, register_pokemon


@register_pokemon
class Squirtle(Pokemon):
    def __init__(self, bot: bool):
        super().__init__(name="杰尼龟",
                         health_point=80,
                         damage=25,
                         defense=20,
                         type_=Type.WATER,
                         evasion_rate=20,
                         skills=
                         [
                             Skill(name="水枪",
                                   action=aqua_jet),
                             Skill(name="护盾",
                                   action=shield),
                         ],
                         bot=bot)

    def on_pre_attacked(self) -> None:
        if random.randint(1, 2) == 1:
            print(f"[{self.name}]通过被动减少了30%的伤害!")
            self.damage_reduction = 30

    def on_post_attacked(self) -> None:
        self.damage_reduction = 0


def aqua_jet(pokemon: Pokemon, enemy: Pokemon) -> None:
    enemy.attacked(pokemon, int(pokemon.damage * 1.4))


def shield(pokemon: Pokemon, enemy: Pokemon) -> None:
    buff = ShieldBuff()
    pokemon.add_buff(buff)
