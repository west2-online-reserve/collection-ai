from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.statuses: list["Effect"] = []
        self.skills = self.initialize_skills()
        self.alive = True
        

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name} uses {skill.name}")
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
        if not isinstance(damage, int):
            damage = int(damage)

        for status in self.statuses:
            if status.name == "Shield":
                 damage = int(damage * (1 - status.amount))
                 print(f"{self.name}'s shield absorbed {int(status.amount*100)}% damage!")
                 status.duration = 0
                 break 
        
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

    def add_status_effect(self, effect: "Effect") -> None:
    # 同名状态直接替换（刷新），避免叠加
        for i, s in enumerate(self.statuses):
            if s.name == effect.name:
                 self.statuses[i] = effect
                 return
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


# GlassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Water":
            effectiveness = 2.0
        elif opponent_type == "Fire":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        # 每个回合开始时执行玻璃属性特性
        self.grass_attribute()

    def grass_attribute(self):
        # 玻璃属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} heals {amount} HP at the start of the turn! Current HP: {self.hp}/{self.max_hp}"
        )

class FirePokemon(Pokemon):
    type = "Fire"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Grass":
            effectiveness = 2.0
        elif opponent_type == "Water":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
       self.after_attribute()

    def after_attribute(self):
        if not hasattr(self,"attack_buff"):
            self.attack_buff=0
        if self.attack_buff<4:
            self.attack_buff+=1
            self.damage=self.attack*(1+0.1*self.attack_buff)
            print(f"{self.name}'s attack increased! Current buff: {self.attack_buff} stacks")

class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Fire":
            effectiveness = 2.0
        elif opponent_type == "Electric":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
       pass

    def after_attribute(self):
        if random.random() < 0.5:
            damage=int(damage * 0.7)
            print(f"{self.name}'s Water attribute reduces damage by 30%!")
            return damage

class ElectricPokemon(Pokemon):
    type = "Electric"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Water":
            effectiveness = 2.0
        elif opponent_type == "Grass":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
       pass

    

# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=35, defense=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=self.attack), skills.ParasiticSeeds(amount=10)]

class PikaChu(ElectricPokemon):
    name = "PikaChu"

    def __init__(self, hp=80, attack=35, defense=5)->None:
        super().__init__(hp,attack,defense)

    def initialize_skills(self):
        return [skills.Thunderbolt(),skills.QuickAttack(damage=self.attack)]

class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20)->None:
        super().__init__(hp,attack,defense)

    def initialize_skills(self):
        return [skills.AquaJet(),skills.Shield()]

class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(self, hp=80, attack=35, defense=15)->None:
        super().__init__(hp,attack,defense)

    def initialize_skills(self):
        return [skills.Ember(damage=35,activation_chance=10),skills.FlameCharge()]

    
