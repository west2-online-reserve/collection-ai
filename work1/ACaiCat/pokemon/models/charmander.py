import random

from enums import Type
from models import Pokemon, Skill, BurnBuff, register_pokemon


@register_pokemon
class Charmander(Pokemon):
    def __init__(self, bot: bool):
        self.attack_times: int = 0
        super().__init__(
            name="小火龙",
            health_point=80,
            damage=35,
            defense=15,
            type_=Type.FIRE,
            evasion_rate=10,
            skills=[
                Skill(name="火花", action=ember),
                Skill(name="蓄能爆炎", action=flame_charge, turns_required=1),
            ],
            bot=bot,
        )

    def on_post_attack(self) -> None:
        if self.attack_times < 4:
            self.damage = int(self.damage * 1.1)
            self.attack_times += 1
            print(
                f"[{self.name}]通过被动提高了10%攻击基础伤害! (当前伤害: {self.damage})"
            )


def ember(pokemon: Pokemon, enemy: Pokemon) -> None:
    hit = enemy.attacked(pokemon, pokemon.damage)

    if hit and random.randint(1, 10) == 1:
        buff = BurnBuff()
        print(f"[{pokemon.name}]对[{enemy.name}]施加了「{buff.name}」")
        enemy.add_buff(buff)


def flame_charge(pokemon: Pokemon, enemy: Pokemon) -> None:
    pokemon.prepared_flame_charge = False
    enemy.miss_bonus += 20
    hit = enemy.attacked(pokemon, pokemon.damage)
    enemy.miss_bonus -= 20
    if hit and random.randint(1, 10) <= 8:
        buff = BurnBuff()
        print(f"[{pokemon.name}]对[{enemy.name}]施加了「{buff.name}」")
        enemy.add_buff(buff)
