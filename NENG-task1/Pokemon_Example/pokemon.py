from __future__ import annotations
import random
import time
import skills
from skills import Skill
from effects import Effect

TYPE_CN = {
    "Grass": "草属性",
    "Fire": "火属性",
    "Water": "水属性",
    "Electric": "电属性",
    "Nai":"奶属性"
}


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int, shanbi: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.shanbi = shanbi
        self.charging_skill = None
        self.has_shield = False
        self.skills = self.initialize_skills()
        self.alive = True
        self.is_paralyzed = False
        self.statuses = []

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        time.sleep(0.5)
        print(f"{self.name} 使用了 {skill.name}！")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        time.sleep(0.5)
        print(f"{self.name} 恢复了 {amount} 点 HP！当前 HP：{self.hp}/{self.max_hp}")

    def receive_damage(self, damage, extra_shanbi=0.0, is_status=False) -> bool:
        if not is_status:
            if random.randint(1, 100) <= min(100, self.shanbi + extra_shanbi):
                time.sleep(0.5)
                print(f"{self.name} 闪避了攻击！")
                return False
            if self.type == "Water" and random.randint(1, 100) <= 50:
                damage *= 0.7
                time.sleep(0.5)
                print(f"{self.name} 触发了水系被动，伤害降低了 30%！")
            if self.has_shield:
                damage *= 0.5
                time.sleep(0.5)
                print(f"{self.name} 的护盾使伤害降低了 50%！")
            damage = int(damage) - self.defense
            if damage <= 0:
                time.sleep(0.5)
                print(f"{self.name} 的防御完全抵挡了本次攻击！")
                return True

        damage = int(damage)
        self.hp = max(0, self.hp - damage)
        time.sleep(0.5)
        print(f"{self.name} 受到了 {damage} 点伤害！剩余 HP：{self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.alive = False
            time.sleep(0.7)
            print(f"{self.name} 昏厥了！")
        return True

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        for existing in self.statuses:
            if type(existing) is type(effect):
                existing.duration = max(existing.duration, effect.duration)
                return
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:
            if not self.alive:
                break
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                time.sleep(0.4)
                print(f"{self.name} 的 {status.name} 效果已结束。")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        self.is_paralyzed = False
        self.has_shield = False
        self.apply_status_effect()

    def __str__(self) -> str:
        return f"{self.name}（{TYPE_CN.get(self.type, self.type)}）"


class GrassPokemon(Pokemon):
    type = "Grass"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        if opponent.type == "Water" :
            effectiveness = 2.0
        elif opponent.type == "Fire"or opponent.type == "Nai":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        super().begin()
        if self.alive:
            self.grass_attribute()

    def grass_attribute(self):
        amount = int(self.max_hp * 0.1)
        self.hp = min(self.max_hp, self.hp + amount)
        time.sleep(0.5)
        print(f"{self.name} 回合开始触发草系被动，恢复了 {amount} 点 HP！当前 HP：{self.hp}/{self.max_hp}")


class FirePokemon(Pokemon):
    type = "Fire"

    def __init__(self, hp: int, attack: int, defense: int, shanbi: int) -> None:
        super().__init__(hp, attack, defense, shanbi)
        self.base_attack = attack
        self.fire_stacks = 0

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Grass":
            return 2.0
        elif opponent.type == "Water"or opponent.type == "Nai":
            return 0.5
        return 1.0

    def use_skill(self, skill: Skill, opponent: Pokemon):
        hp_before = opponent.hp
        super().use_skill(skill, opponent)
        if opponent.hp < hp_before and self.fire_stacks < 4:
            self.fire_stacks += 1
            self.attack = int(self.base_attack * (1 + 0.1 * self.fire_stacks))
            time.sleep(0.5)
            print(f"{self.name} 触发了火系被动！攻击力提升至 {self.attack}（{self.fire_stacks}/4 层）")


class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Fire":
            return 2.0
        elif opponent.type == "Electric" or opponent.type == "Nai":
            return 0.5
        return 1.0


class ElectricPokemon(Pokemon):
    type = "Electric"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Water":
            return 2.0
        elif opponent.type == "Grass" or opponent.type == "Nai":
            return 0.5
        return 1.0


class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"

    def __init__(self, hp=100, attack=35, defense=10, shanbi=10) -> None:
        super().__init__(hp, attack, defense, shanbi)

    def initialize_skills(self):
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class Pikachu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self, hp=80, attack=35, defense=5, shanbi=30) -> None:
        super().__init__(hp, attack, defense, shanbi)

    def initialize_skills(self):
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, hp=80, attack=25, defense=20, shanbi=20) -> None:
        super().__init__(hp, attack, defense, shanbi)

    def initialize_skills(self):
        return [skills.AquaJet(), skills.Shield()]


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self, hp=80, attack=35, defense=15, shanbi=10) -> None:
        super().__init__(hp, attack, defense, shanbi)

    def initialize_skills(self):
        return [skills.Ember(), skills.FlameCharge()]

    
class Naiwa(Pokemon):
    name = "奶蛙"
    type = "Nai"
    def __init__(self,hp = 100,attack = 40 , defense = 20,shanbi = 20 ):
        super().__init__(hp, attack, defense, shanbi)
        self.numpie = 0
    def type_effectiveness(self, opponent: Pokemon) -> float:
           
            return 0.5
    def initialize_skills(self):
            return [skills.pengfu(), skills.yipie()]