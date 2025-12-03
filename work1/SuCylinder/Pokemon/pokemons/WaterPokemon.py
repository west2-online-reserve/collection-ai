from base.pokemon import Pokemon
from skills import SquirtleSkills
from misc.tools import printWithDelay
import random


class WaterPokemon(Pokemon):
    type = "水"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "火":
            printWithDelay("效果拔群!")
            effectiveness = 2.0
        elif opponent_type == "电":
            printWithDelay("收效甚微")
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        self.apply_status_effect()

    def receive_damage(self, damage, type, chance=50, damage_reduction=0.3):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        if type not in self.effect_list:
            damage -= self.defense
            if damage <= 0:
                printWithDelay(f"{self.name} 防御了这次攻击!")
                return
        if random.randint(1, 100) <= chance:
            printWithDelay(f"{self.name}触发伤害减免!")
            damage *= 1 - damage_reduction - self.damage_reduction
        else:
            damage *= 1 - self.damage_reduction
        damage = round(damage)
        self.hp -= damage
        printWithDelay(f"{self.name} 受到了 {type} 的 {damage} 点伤害!", end=" ")
        printWithDelay(f"当前 HP: {self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.alive = False
            printWithDelay(f"{self.name} 倒下了!")


class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, hp=80, attack=25, defense=20, dodge_chance=20) -> None:
        super().__init__(hp, attack, defense, dodge_chance)

    def initialize_skills(self):
        return [SquirtleSkills.Aqua_Jet(), SquirtleSkills.Shield()]
