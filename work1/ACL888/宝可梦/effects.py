
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon
class Effect:
    name:str

    def __init__ (self,duration:int)->None:
        self.duration = duration

    def apply(self,pokemon:"Pokemon")->None:
        raise NotImplementedError
     
    def decrease_duration(self)->None:
       self.duration-=1
       print(f"{self.name} effect duration decreased.Remaining:{self.duration}")

    def on_expire(self, pokemon: "Pokemon") -> None:
        """Optional hook called when an effect expires so it can revert changes on the pokemon."""
        pass

class PoisonEffect(Effect):
    name = "Poison"

    def __init__(self,damage:int=10,duration:int=3)->None:
        super().__init__(duration)
        self.damage = damage

    def apply(self,pokemon)->None:
        # Poison deals raw damage that bypasses defense
        pokemon.hp -= self.damage
        if pokemon.hp <= 0:
            pokemon.alive = False
        print(f"{pokemon.name} takes {self.damage} poison damage")

class HealEffect(Effect):
    name = "Heal"

    def __init__(self,amount:int,duration:int=3)->None:
        super().__init__(duration)
        self.amount = amount

    def apply(self,pokemon:"Pokemon")->None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} heals {self.amount} HP!")
    
class burnEffect(Effect):
    name = "Burn"

    def __init__(self, damage:int=10,duration:int=2)->None:
        super().__init__(duration)
        self.damage = damage

    def apply(self,pokemon:"Pokemon")->None:
        # Burn deals raw damage ignoring defense
        pokemon.hp -= self.damage
        if pokemon.hp <= 0:
            pokemon.alive = False
        print(f"{pokemon.name} takes {self.damage} burn damage")
    def decrease_duration(self) -> None:
        super().decrease_duration()
class ParalyzeEffect(Effect):
    # 麻痹Effect(Effect)，amount 可保留作为潜在强度/占位参数
    name = "Paralyze"
    def __init__(self, amount: int = 1, duration: int = 1) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self,pokemon:"Pokemon")->None:
        pokemon.paralyzed = True
        print(f"{pokemon.name} is paralyzed and cannot move!")

    def decrease_duration(self) -> None:
        super().decrease_duration()

    def on_expire(self, pokemon: "Pokemon") -> None:
        pokemon.paralyzed = False
        print(f"{pokemon.name} is no longer paralyzed.")
class ChargeEffect(Effect):
    name = "Charge"
    # 蓄力准备技能,让Pokemon待机一回合

    def __init__(self, amount: int = 1, damage: int = 0, target=None, duration: int = 1) -> None:
        super().__init__(duration)
        self.amount = amount
        self.damage = damage
        # store a (weak) reference to target for when charge releases
        self._target = target
        self._applied = False
    def apply(self, pokemon: "Pokemon") -> None:
        # 如果已经在蓄力中，不重复叠加
        if getattr(pokemon, "charging", False):
            return
        pokemon.charging = True
        if not hasattr(pokemon, "charge"):
            pokemon.charge = 0
        pokemon.charge += self.amount
        self._applied = True
        print(f"{pokemon.name} charges {self.amount} energy!")
    def decrease_duration(self)->None:
        super().decrease_duration()
    def on_expire(self, pokemon: "Pokemon") -> None:
        # 当 effect 到期时释放蓄力（对预设目标造成伤害），并清理状态
        pokemon.charging = False
        print(f"{pokemon.name} has finished charging.")
        if self._applied and self.damage and self._target is not None:
            try:
                self._target.receive_damage(self.damage)
                print(f"{pokemon.name}'s charged attack deals {self.damage} damage to {self._target.name}!")
            except Exception:
                # 如果目标不存在/已切换/异常，忽略
                pass


class DefenseEffect(Effect):
    name = "Defense"

    def __init__(self, amount: int = 10, duration: int = 2) -> None:
        super().__init__(duration)
        self.amount = amount
        self._applied = False

    def apply(self, pokemon: "Pokemon") -> None:
        # 第一次应用时提高防御
        if not self._applied:
            pokemon.defense += self.amount
            self._applied = True
            print(f"{pokemon.name} gains {self.amount} defense from Shield.")

    def on_expire(self, pokemon: "Pokemon") -> None:
        if self._applied:
            pokemon.defense -= self.amount
            self._applied = False
            print(f"{pokemon.name}'s shield has worn off, -{self.amount} defense.")
