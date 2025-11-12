from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random # 新增：导入random模块


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int, dodge_rate: float = 0.0) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.dodge_rate = dodge_rate # 新增：添加闪避率
        self.skills = self.initialize_skills()
        self.alive = True
        # self.statuses = []
        self.current_effects = []

    # 新增：判断属性克制
    def get_type_effectiveness(self, skill_type: str, opponent_type: str) -> float:
        effectiveness_chart = {
            "Grass": {"Water": 2.0, "Fire": 0.5},
            "Fire": {"Grass": 2.0, "Water": 0.5},
            "Water": {"Fire": 2.0, "Electric":0.5},
            "Electric": {"Water":2.0, "Grass": 0.5}
        }

        # effectiveness_chart：查外层字典的键
        # effectiveness_chart[skill_type]: 查内层字典的键
        # effectiveness_chart[skill_type][opponent_type]：查对应的克制值
        if skill_type in effectiveness_chart and opponent_type in effectiveness_chart[skill_type]:
            return effectiveness_chart[skill_type][opponent_type]
        else:
            return 1.0

    # 新增：计算伤害
    def calculate_damage(self, user_type: str, opponent_type: str, user: "Pokemon", opponent: "Pokemon") -> int:
        effectiveness = self.get_type_effectiveness(user_type, opponent_type)

        attack_damage = int(user.attack * effectiveness)
        actual_damage = attack_damage - opponent.defense
        actual_damage = max(1, actual_damage)

        return actual_damage

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name}使用了{skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage1):
        if not isinstance(damage1, int):
            damage1 = int(damage1)

        # for effect in self.current_effects[:]: # [:]是切片语法，可以创建列表的完整副本
        #     if hasattr(effect, "on_receive_damage"): # hasattr(对象，“属性名”)：检查effect对象是否有on_receive_damage方法
        #         actual_damage = effect.on_receive_damage(damage, self)
        actual_damage = self.apply_damage_effects(damage1)

        # 更新hp
        self.hp -= actual_damage
        print(f"{self.name}受到了{actual_damage}点伤害，当前血量：{self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")


    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.current_effects.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.current_effects[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.current_effects.remove(status)

    def apply_damage_effects(self, damage1):
        actual_damage = damage1
        for status in self.current_effects[:]:  # 使用切片防止列表在遍历时被修改
            if hasattr(status, "on_receive_damage"):  # hasattr(对象，“属性名”)：检查effect对象是否有on_receive_damage方法
                actual_damage = status.on_receive_damage(damage1)

        return actual_damage

    # def type_effectiveness(self, opponent: Pokemon):
    #     # 计算属性克制的抽象方法，具体实现由子类提供
    #     raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"

# GrassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

    # def type_effectiveness(self, opponent: Pokemon):
    #     # 针对敌方 Pokemon 的类型，调整效果倍率
    #     effectiveness = 1.0
    #     opponent_type = opponent.type
    #
    #     if opponent_type == "Water":
    #         effectiveness = 2.0
    #     elif opponent_type == "Fire":
    #         effectiveness = 0.5
    #     return effectiveness

    def begin(self):
        # 每个回合开始时执行草属性特性
        self.apply_status_effect()
        self.grass_attribute()

    def grass_attribute(self):
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} heals {amount} HP at the start of the turn! Current HP: {self.hp}/{self.max_hp}"
        )

# 新增：FirePokemon类
class FirePokemon(Pokemon):
    type = "Fire"

    # # 针对敌方 Pokemon 的类型，调整效果倍率
    # def type_effectiveness(self, opponent: Pokemon) -> float:
    #     return self.get_type_effectiveness("Fire", opponent.type)

    def begin(self):
        self.apply_status_effect()
        self.fire_attribute()

    # 火属性特性：每次造成伤害，叠加10%的攻击力，最多4层
    def fire_attribute(self):
        amount = self.attack * 0.1
        self.attack += amount
        if self.attack > self.attack + amount * 4:
            self.attack = self.attack + amount * 4

# 新增：WaterPokemon类
class WaterPokemon(Pokemon):
    type = "Water"

    # # 针对敌方 Pokemon 的类型，调整效果倍率
    # def type_effectiveness(self, opponent: Pokemon) -> float:
    #     return self.get_type_effectiveness("Water", opponent.type)

    def begin(self):
        self.apply_status_effect()

    # 水属性被动：受到伤害时，有50%的几率减免30%的伤害
    def water_attribute(self, damage):
        if random.random() < 0.5:
            reduced_damage = damage * 0.7
            return reduced_damage
        return damage

    def apply_damage_effects(self, damage1):
        # 初始化实际伤害
        actual_damage = damage1

        # 先应用状态效果
        for status in self.current_effects[:]:  # 使用切片防止列表在遍历时被修改
            if hasattr(status, "on_receive_damage"):  # hasattr(对象，“属性名”)：检查effect对象是否有on_receive_damage方法
                actual_damage = status.on_receive_damage(damage1)

        # 再应用水属性被动
        actual_damage = self.water_attribute(actual_damage)

        # 返回应用完状态效果和被动后的实际伤害
        return actual_damage

# 新增：ElectricPokemon类
class ElectricPokemon(Pokemon):
    type = "Electric"

    # # 针对敌方 Pokemon 的类型，调整效果倍率
    # def type_effectiveness(self, opponent: Pokemon) -> float:
    #     return self.get_type_effectiveness("Electric", opponent.type)

    # 电属性被动：当成功躲闪时，可以立即使用一次技能
    def use_skill_again(self):
        if self.skills:
            skill = random.choice(self.skills)
            print(f"触发电属性被动，立即使用{skill.name}")


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=35, defense=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds()]

# 新增：PikaChu类
class PikaChu(ElectricPokemon):
    name = "PikaChu"

    def __init__(self, hp=80, attack=35, defense=5) -> None:
        # 初始化PikaChu的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能
        return [skills.Thunderbolt(), skills.QuickAttack()]

# 新增：Squirtle类
class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20) -> None:
        # 初始化Squirtle的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能
        return [skills.AquaJet(), skills.Shield()]

# 新增：Charmander类
class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(self, hp=80, attack=35, defense=15) -> None:
        # 初始化Charmander的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能
        return [skills.Ember(), skills.FlameCharge()]