from enums import Type
from models import Pokemon, Skill, register_pokemon


@register_pokemon
class HaCat(Pokemon):
    def __init__(self, bot: bool):
        super().__init__(
            name="哈基米",
            health_point=40,
            damage=15,
            defense=5,
            type_=Type.DARK,
            evasion_rate=60,
            skills=[
                Skill(name="猫猫普攻", action=scratch),
                Skill(name="哈气", action=ha, turns_required=3),
            ],
            bot=bot,
        )


def scratch(pokemon: Pokemon, enemy: Pokemon) -> None:
    enemy.attacked(pokemon, int(pokemon.damage * 1.4))


def ha(pokemon: Pokemon, enemy: Pokemon) -> None:
    enemy.attacked(pokemon, 50)
    pokemon.ha_progress = 0
