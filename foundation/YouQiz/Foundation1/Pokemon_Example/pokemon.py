"""宝可梦基类、属性子类与图鉴。"""

from __future__ import annotations

import skills
from effects import Effect
from skills import Skill


class Pokemon:
    name: str
    type: str

    def __init__(
        self, hp: int, attack: int, defense: int, dodge: float = 0.0
    ) -> None:
        self.hp = hp
        self.max_hp = hp
        self.base_attack = attack
        self.attack = attack
        self.defense = defense
        self.dodge = dodge
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses: list[Effect] = []
        self.charging_skill: Skill | None = None
        self.skip_action_reason: str | None = None

    def initialize_skills(self) -> list[Skill]:
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon) -> None:
        print(f"{self.name} uses {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount: int) -> None:
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 回复了 {amount} 点 HP！当前 HP：{self.hp}/{self.max_hp}")

    def receive_damage(self, damage: int) -> None:
        damage = int(damage) - self.defense
        if damage <= 0:
            print(f"{self.name} 的防御抵挡了攻击！")
            return

        self.hp -= damage
        if self.hp < 0:
            self.hp = 0
        print(f"{self.name} 受到了 {damage} 点伤害！剩余 HP：{self.hp}")
        if self.hp == 0:
            self.alive = False
            print(f"{self.name} 倒下了！")

    def add_status_effect(self, effect: Effect) -> None:
        if not self.alive:
            return
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        for status in self.statuses[:]:
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name} 的「{status.name}」效果消失了。")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon) -> float:
        raise NotImplementedError

    def begin(self) -> None:
        self.apply_status_effect()

    def __str__(self) -> str:
        return f"{self.name}（{self.type}属性）"


class GrassPokemon(Pokemon):
    type = "草"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type in {"水", "电"}:
            return 2.0
        if opponent.type == "火":
            return 0.5
        return 1.0

    def begin(self) -> None:
        super().begin()
        if not self.alive or self.hp >= self.max_hp:
            return
        amount = int(self.max_hp * 0.10)
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} 在回合开始时回复了 {amount} 点 HP！"
            f"当前 HP：{self.hp}/{self.max_hp}"
        )


class FirePokemon(Pokemon):
    type = "火"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "草":
            return 2.0
        if opponent.type == "水":
            return 0.5
        return 1.0


class WaterPokemon(Pokemon):
    type = "水"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "火":
            return 2.0
        if opponent.type in {"草", "电"}:
            return 0.5
        return 1.0


class ElectricPokemon(Pokemon):
    type = "电"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "水":
            return 2.0
        if opponent.type == "草":
            return 0.5
        return 1.0


class Pikachu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self) -> None:
        super().__init__(80, 35, 5, 0.30)

    def initialize_skills(self) -> list[Skill]:
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"

    def __init__(self) -> None:
        super().__init__(100, 50, 10, 0.10)

    def initialize_skills(self) -> list[Skill]:
        # 种子炸弹按草属性伤害计算，寄生种子提供回复和中毒
        return [skills.SeedBomb(), skills.ParasiticSeeds(amount=10)]

class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self) -> None:
        super().__init__(80, 25, 20, 0.20)

    def initialize_skills(self) -> list[Skill]:
        return [skills.AquaJet(), skills.Shield()]


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self) -> None:
        super().__init__(80, 35, 15, 0.10)

    def initialize_skills(self) -> list[Skill]:
        return [skills.Ember(), skills.FlameCharge()]


class Raichu(ElectricPokemon):
    name = "雷丘"

    def __init__(self) -> None:
        super().__init__(85, 40, 8, 0.25)

    def initialize_skills(self) -> list[Skill]:
        return [skills.Thunderbolt(), skills.ThunderPunch()]


class Blastoise(WaterPokemon):
    name = "水箭龟"

    def __init__(self) -> None:
        super().__init__(95, 30, 25, 0.15)

    def initialize_skills(self) -> list[Skill]:
        return [skills.HydroPump(), skills.Shield()]


POKEDEX: tuple[Pokemon, ...] = (
    Pikachu(),
    Bulbasaur(),
    Squirtle(),
    Charmander(),
    Raichu(),
    Blastoise(),
)