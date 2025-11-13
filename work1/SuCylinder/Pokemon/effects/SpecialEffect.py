from base.effects import Effect
from base.pokemon import Pokemon
from skills import CharmanderSkills
from misc.tools import printWithDelay


class DamageReductionEffect(Effect):
    name = "伤害减免"

    def __init__(self, amount: float = 0.5, duration: int = 1) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon):
        if self.duration > 0:
            printWithDelay(f"{pokemon.name} 现在拥有 {self.amount*100}% 的伤害减免")
            pokemon.damage_reduction = self.amount

    def effect_clear(self, pokemon):
        pokemon.damage_reduction = 0


class Flame(Effect):
    name = "蓄力中"

    def __init__(self, opponent: "Pokemon", duration: int = 2) -> None:
        super().__init__(duration)
        self.target = opponent

    def apply(self, pokemon):
        if self.duration > 1:
            printWithDelay(f"{pokemon.name} 蓄力中,无法行动")
            pokemon.cant_move = True

    def effect_clear(self, pokemon):
        pokemon.cant_move = False
        printWithDelay(f"{pokemon.name} 蓄力完成")
        pokemon.use_skill(CharmanderSkills.Flame_Charge_fire(), self.target)


class VampiricEffect(Effect):
    name = "寄生种子"

    def __init__(self, opponent: "Pokemon", amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount
        self.target = opponent

    def apply(self, pokemon: "Pokemon") -> None:
        damage = self.target.get_max_hp() * self.amount
        if self.target.alive:
            printWithDelay(
                f"{pokemon.name} 偷取了{self.target.name} 的 {damage} 点 HP!"
            )
            self.target.receive_damage(damage, self.name)
            pokemon.heal_self(damage)
        else:
            self.duration = -1
