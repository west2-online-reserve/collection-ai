from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random

class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int,dodge_rate: float = 0.0) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills = self.initialize_skills()
        self.alive = True
        self.dodge_rate = dodge_rate
        self.statuses = []
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

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError
    
    #这个方法为所有宝可梦子类定义了一个标准接口：每个宝可梦都应该有回合开始时的行为。
    def begin(self):
        # 新回合开始时触发的方法
        pass\
    

    def try_dodge(self) -> bool:
        """尝试闪避攻击"""
        if random.random() <= self.dodge_rate:
            print(f"{self.name} 成功闪避了攻击！")
            return True
        return False

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

class ElectricPokemon(Pokemon):
    type = "Electric"

    def type_effectiveness(self, opponent:Pokemon) :
        """计算属性克制倍率"""
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Grass":
            effectiveness = 0.5
        elif opponent_type == "Fire":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        """每个回合开始时执行电属性特性"""
        super().begin()
        # 电属性特性在 try_dodge 中实现
    
    def electric_attribute(self):
        """电属性特性：高闪避率并且在闪避后可能获得额外行动机会"""
        # 电属性的主要特性已经在 try_dodge 中实现
        # 这里可以添加其他电属性相关的回合开始效果
        pass
    
    def try_dodge(self) -> bool:
        """电属性特性：高闪避率，闪避后可能获得反击机会"""
        if random.random() <= self.dodge_rate:
            print(f"{self.name} 成功闪避了攻击！")
            
            # 电属性特性：30%几率在闪避后立即反击
            if random.random() <= 0.3:
                print(f"{self.name} 的电属性特性触发，获得反击机会！")
                # 这里需要战斗系统的支持来实际执行反击
                # 暂时标记，在战斗逻辑中处理
                self.has_counter_attack = True
            return True
        return False
class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon):
        """计算属性克制倍率"""
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Fire":
            effectiveness = 2.0
        elif opponent_type == "Grass":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        """每个回合开始时执行水属性特性"""
        super().begin()
        self.water_attribute()

    def water_attribute(self):
        """水属性特性：受到攻击时有几率减少伤害"""
        # 水属性的主要特性在防御时体现
        pass
    
    def receive_damage(self, damage: int) -> None:
        """水属性特性：有几率减少受到的伤害"""
        # 20%几率减少30%伤害
        if random.random() <= 0.2:
            reduced_damage = int(damage * 0.7)
            super().receive_damage(reduced_damage)
            print(f"{self.name}的水属性特性触发，减少了30%伤害！")
        else:
            super().receive_damage(damage)


# 杰尼龟类，继承自WaterPokemon
class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20, dodge_rate=0.2) -> None:
        # 初始化杰尼龟的属性
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        # 初始化技能，具体技能是AquaJet和Shield
        return [skills.AquaJet(damage=140), skills.Shield()]


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=50, defense=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]

 
class Pikachu(ElectricPokemon):
    name = "Pikachu"

    def __init__(self, hp=80, attack=35, defense=5, dodge_rate=0.3) -> None:
        # 初始化 Pikachu 的属性
        super().__init__( hp, attack, defense, dodge_rate)
        self.skills = self.initialize_skills()
        self.has_counter_attack = False  # 电属性反击标记

    def initialize_skills(self):
        # 初始化技能，具体技能是 Thunderbolt 和 QuickAttack
        return [skills.Thunderbolt(damage=10), skills.QuickAttack(damage=10)]
    
    def begin(self):
        """皮卡丘回合开始处理"""
        super().begin()
        # 重置反击标记
        self.has_counter_attack = False

    def get_counter_attack_skill(self):
        """获取用于反击的技能"""
        if self.skills:
            return self.skills[0]  # 使用第一个技能进行反击
        return None
    
