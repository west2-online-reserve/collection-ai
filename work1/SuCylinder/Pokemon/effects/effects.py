from base.effects import Effect
from base.pokemon import Pokemon
from misc.tools import printWithDelay


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, amount: float = 0.1, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        damage = pokemon.get_max_hp() * self.amount
        pokemon.receive_damage(damage, self.name)


class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, amount: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon):
        damage = self.amount
        pokemon.receive_damage(damage, self.name)


class ParalysisEffect(Effect):
    name = "麻痹"

    def __init__(self, duration: int = 2) -> None:
        super().__init__(duration)

    def apply(self, pokemon):
        if self.duration > 1:
            printWithDelay(f"{pokemon.name} 被麻痹了,无法行动")
            pokemon.cant_move = True

    def effect_clear(self, pokemon):
        pokemon.cant_move = False
