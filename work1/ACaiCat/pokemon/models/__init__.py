from typing import Type
from .skill import Skill
from .buff import Buff, BurnBuff, ParalysisBuff, ShieldBuff, ParasiticBuff, PoisonBuff
from .pokemon import Pokemon

_pokemon_registry: list[Type[Pokemon]] = []


def register_pokemon(cls: Type[Pokemon]):
    _pokemon_registry.append(cls)
    return cls


def get_all_pokemons() -> list[Type[Pokemon]]:
    return _pokemon_registry.copy()


from .bulbasaur import Bulbasaur
from .charmander import Charmander
from .ha import HaCat
from .pikachu import PikaChu
from .squirtle import Squirtle
