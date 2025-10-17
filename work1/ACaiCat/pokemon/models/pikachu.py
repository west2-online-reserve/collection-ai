import random

from enums import Type
from models import Pokemon, Skill, ParalysisBuff, register_pokemon


@register_pokemon
class PikaChu(Pokemon):
    def __init__(self, bot: bool):
        super().__init__(name="皮卡丘",
                         health_point=80,
                         damage=35,
                         defense=5,
                         type_=Type.ELECTRIC,
                         evasion_rate=30,
                         skills=
                         [
                             Skill(name="十万伏特",
                                   action=thunderbolt),
                             Skill(name="电光一闪",
                                   action=quick_attack),
                         ],
                         bot=bot)

    def on_mise_attack(self) -> None:
        print(f"[{self.name}]触发被动额外施展一次技能!")
        self.select_skill()


def thunderbolt(pokemon: Pokemon, enemy: Pokemon) -> None:
    damage = int(pokemon.damage * 1.4)
    enemy.attacked(pokemon, damage)

    if random.randint(1, 10) == 1:
        buff = ParalysisBuff()
        print(f"[{pokemon.name}]对[{enemy.name}]施加了「{buff.name}」")
        enemy.add_buff(buff)


def quick_attack(pokemon: Pokemon, enemy: Pokemon) -> None:
    enemy.attacked(pokemon, pokemon.damage)
    if random.randint(1, 10) == 1:
        enemy.attacked(pokemon, pokemon.damage)
