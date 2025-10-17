import random

from enums import Type
from models import Pokemon, Skill, PoisonBuff, ParasiticBuff, register_pokemon


@register_pokemon
class Bulbasaur(Pokemon):
    def __init__(self, bot: bool):
        super().__init__(
            name="妙蛙种子",
            health_point=100,
            damage=35,
            defense=10,
            type_=Type.GRASS,
            evasion_rate=10,
            skills=[
                Skill(name="种子炸弹", action=seed_bomb),
                Skill(name="寄生种子", action=parasitic_seeds),
            ],
            bot=bot,
        )

    def on_turn_start(self) -> None:
        recovery = int(self.max_health_point / 10)
        print(
            f"[{self.name}]通过被动恢复了{recovery}点生命值! (剩余HP: {self.health_point})"
        )
        self.health_point += recovery


def seed_bomb(pokemon: Pokemon, enemy: Pokemon) -> None:
    hit = enemy.attacked(pokemon, pokemon.damage)

    if hit and random.randint(1, 100) <= 15:
        buff = PoisonBuff()
        print(f"[{pokemon.name}]对[{enemy.name}]施加了{buff.name}")
        enemy.add_buff(buff)


def parasitic_seeds(pokemon: Pokemon, enemy: Pokemon) -> None:
    buff = ParasiticBuff()
    pokemon.add_buff(buff)
