from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Pokemon


class Skill:
    def __init__(self, name: str, action: Callable[["Pokemon", "Pokemon"], None], turns_required: int = 0):
        self.name: str = name
        self.action: Callable[["Pokemon", "Pokemon"], None] = action
        self.turns_required = turns_required
        self.current_turns: int = 0

    def perform(self, pokemon: "Pokemon", enemy: "Pokemon"):
        self.action(pokemon, enemy)
