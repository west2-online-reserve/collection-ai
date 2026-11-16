from __future__ import annotations
import skills
import random
from skills import Skill
from effects import Effect

TYPE_ADVANTAGE = {
    "Water": ["Fire"],
    "Fire": ["Grass"],
    "Grass": ["Water"],
    "Electric": ["Water"],
    "Ground": ["Electric"]
}

class Pokemon:
    name: str
    type: str
    dodge_rate: float = 0.0

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses = []
        self.shield = False
        self.charging_skill = None
        self.attack_stage = 0

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def calculate_outgoing_damage(self, base, opponent, move_type):
        eff = self.type_effectiveness(move_type, opponent)
        dmg = int(base * (1 + 0.1 * self.attack_stage) * eff)

        return dmg

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name} uses {skill.name}")
        #麻痹效果
        for s in self.statuses:
            if s.name == "Paralysis" and random.random() < 0.3:
                print(f"{self.name} is paralyzed and can't move!")
                return
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        damage = self.modify_incoming_damage(damage)
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return

        self.hp -= damage
        print(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)

    def type_effectiveness(self, move_type: str, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        if opponent.type in TYPE_ADVANTAGE.get(move_type, []):
            return 2.0
        if move_type in TYPE_ADVANTAGE.get(opponent.type, []):
            return 0.5
        return 1.0

    def on_deal_damage(self):
        pass

    def on_dodge(self, opponent):
        pass

    def modify_incoming_damage(self, damage):
        return damage

    def begin(self):
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


# GlassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

    def begin(self):
        heal = int(self.max_hp * 0.1)
        self.hp = min(self.max_hp, self.hp + heal)
        print(f"[Grass] {self.name} heals {heal} HP! ({self.hp}/{self.max_hp})")

class FirePokemon(Pokemon):
    type = "Fire"

    def __init__(self, hp, attack, defense):
        super().__init__(hp, attack, defense)
        self.attack_stage = 0

    def on_deal_damage(self):
        if self.attack_stage < 4:
            self.attack_stage += 1
            print(f"[Fire] {self.name} attack +10% (stack {self.attack_stage})")

class WaterPokemon(Pokemon):
    type = "Water"

    def modify_incoming_damage(self, damage):
        if random.random() < 0.5:
            print(f"[Water] {self.name} reduces damage by 30%!")
            return int(damage * 0.7)
        return damage


class ElectricPokemon(Pokemon):
    type = "Electric"

    def on_dodge(self, opponent):
        print(f"[Electric] {self.name} dodged and counterattacks!")
        if self.skills:
            self.skills[0].execute(self, opponent)

# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp: int = 100, attack: int = 50, defense: int = 10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]

class PikaChu(ElectricPokemon):
    name = "Pikachu"
    dodge_rate = 0.3

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 5):
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Squirtle(WaterPokemon):
    name = "Squirtle"
    dodge_rate = 0.2

    def __init__(self, hp: int = 80, attack: int = 25, defense: int = 20):
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        return [skills.AquaJet(), skills.Shield()]

class Charmander(FirePokemon):
    name = "Charmander"
    dodge_rate = 0.1

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 15):
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        return [skills.Ember(), skills.FlameCharge()]

class Sandshrew(Pokemon):
    name = "Sandshrew"
    type = "Ground"
    dodge_rate = 0.15

    def __init__(self, hp: int = 90, attack: int = 15, defense: int = 25):
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        return [skills.Dig(), skills.SandAttack()]