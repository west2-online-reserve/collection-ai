from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random
from color import *

class Pokemon:
    name: str
    type: str

    def __init__(self, master:str, hp: int, attack: int, defense: int, dodge_rate: int) -> None:
        # 初始化 Pokemon 的属性
        self.master = master
        self.hp = hp                                # 当前血量
        self.max_hp = hp                            # 最大血量
        self.attack = attack                        # 攻击
        self.defense = defense                      # 防御
        self.skills = self.initialize_skills()      # 技能
        self.alive = True                           # 生存状态
        self.statuses = []                          # 状态效果
        self.dodge_rate = dodge_rate                  # 躲闪率
        self.last_round_dodged = False             # 电属性专用
        self.paralyzed = False                    # 麻痹属性(针对电属性的麻痹攻击)
        self.passive_defense_rate = 0             # 被动防御概率
        self.passive_defense_success = False      # 被动防御成功
        self.protective_shield = False           # 保护罩开启
        self.current_skill = None                # 当前使用技能
        self.super_position_times = -1           # 攻击叠加次数(考虑到第一回合不存在叠加次数，所以设置成负数好弄)

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def distinguish_name(self, player_color, computer_color):
        if self.master == POSITION["player"]:
            self.name = color(self.name, player_color)
        elif self.master == POSITION["opponent"]:
            self.name = color(self.name, computer_color)

    def passive_defense(self):
        if random.randint(1, 100) <= self.passive_defense_rate:
            self.passive_defense_success = True
            return True

        self.passive_defense_success = False
        return False

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        self.current_skill = skill
        if skill.can_release:
            print(f"{self.name} 使用 {skill.name}")
            skill.execute(self, opponent)
            return True
        else:
            print(f"{self.master} :{skill.name} 冷却中……")
            return False

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 恢复 {amount} HP! 当前 HP: {self.hp}/{self.max_hp}")

    def dodge_success(self, add_dodge = 0):
        if random.randint(1, 100) <= self.dodge_rate + add_dodge:
            self.last_round_dodged = True
            return True
        self.last_round_dodged = False
        return False

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name} 完全防御了这次攻击!")
            return

        self.hp -= damage
        self.hp = max(self.hp, 0)
        print(
            f"{self.name} 受到 {damage} 点伤害! 剩余 HP: {self.hp}/{self.max_hp}"
        )
        if self.hp == 0:
            self.alive = False
            print(f"{self.name} 晕厥!")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改（复制了一份，保护了原列表）
            status.apply(self)
            status.decrease_effect_round()
            if status.effect_disappear():
                print(f"{self.name} 的 {status.name} 状态效果消退。")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self, opponent: Pokemon):
        # 新回合开始时触发的方法
        self.paralyzed = False              # 每回合起始麻痹状态消失
        self.apply_status_effect()
        if self.current_skill is not None:
           for skill in self.skills:
               skill.decrease_cool_round()

    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"


# GlassPokemon 类, 继承自 Pokemon 类
class GrassPokemon(Pokemon):
    type = "草"              # 类属性，实例共享，不属于单独对象

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0     # 效果倍率
        opponent_type = opponent.type

        if opponent_type == "水":
            effectiveness = 2.0
             
        elif opponent_type == "火":
            effectiveness = 0.5

        return effectiveness

    def begin(self, opponent: Pokemon):
        # 每个回合开始时执行玻璃属性特性
        self.grass_attribute()
        super().begin(self)

    def grass_attribute(self):
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} 恢复了 {amount} HP! 当前 HP: {self.hp}/{self.max_hp}"
        )

# Bulbasaur 类，继承自 GrassPokemon    妙蛙种子
class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"    # 默认 type 为 GrassPokemon 的 type

    def __init__(self, master, hp=100, attack=50, defense=10, dodge_rate = 10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(master, hp, attack, defense, dodge_rate)    # 调用父类的构造函数

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]


class ElectricPokemon(Pokemon):
    type = "电"
    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type


        if opponent_type == "水":
            effectiveness = 2.0
             
        if opponent_type == "草":
            effectiveness = 0.5
              
        return effectiveness

    def begin(self, opponent: Pokemon):
        if self.last_round_dodged:
            skill = random.choice(self.skills)
            print(f"上一回合 {self.name} 成功闪避, 本回合随机触发技能 {skill.name} !")
            skill.execute(self, opponent)
        super().begin(self)


class PikaChu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self, master, hp = 80, attack = 35, defense = 5, dodge_rate = 30) -> None:
        super().__init__(master, hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        return [skills.Thunderbolt(damage = self.attack), skills.QuickAttack(damage = self.attack)]


class WaterPokemon(Pokemon):
    type = "水"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0     # 效果倍率
        opponent_type = opponent.type

        if opponent_type == "火":
            effectiveness = 2.0
             
        elif opponent_type == "电":
            effectiveness = 0.5

        return effectiveness

    def begin(self, opponent: Pokemon):
        if super().passive_defense():
            print(f"本回合 {self.name} 成功触发防御被动，伤害减免 30% !")
        super().begin(self)

    def water_attribute(self):
        pass


class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, master, hp=80, attack=25, defense=20, dodge_rate= 20) -> None:
        super().__init__(master, hp, attack, defense, dodge_rate)
        self.passive_defense_rate = 50

    def initialize_skills(self):
        return [skills.Aquajet(damage = self.attack), skills.Shield(0)]


class FirePokemon(Pokemon):
    type = "火"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type
        if opponent_type == "草":
            effectiveness = 2.0
             
        elif opponent_type == "水":
            effectiveness = 0.5

        return effectiveness

    def begin(self, opponent: Pokemon):
        super().begin(self)
        if not opponent.last_round_dodged:
            if 0 <= self.super_position_times < 4:    # 叠加次数正常范围
                self.super_position_times += 1
                print(
                    f"{self.name} 连续 {self.super_position_times} 次成功构成了伤害,本回合攻击提升 {self.super_position_times * 10}%!")
            elif self.super_position_times < 0:       # 刚开局
                self.super_position_times = 0
            else:                                    # 叠加次数大于 4 了，但是依然构成连续攻击，则从 1 开始
                self.super_position_times = 1
                print(
                    f"{self.name} 连续 {self.super_position_times} 次成功构成了伤害,本回合攻击提升 {self.super_position_times * 10}%!")
        else:
            self.super_position_times = 0


    def fire_attribute(self):
        pass


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self, master, hp=80, attack=35, defense=15, dodge_rate= 10) -> None:
        super().__init__(master, hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        return [skills.Ember(damage = self.attack), skills.Flamecharge(damage = self.attack)]

class BlankPokemon(Pokemon):
    name = "空白之物"
    type = "空白"

    def __init__(self, master, hp = 1000000, attack = 0, defense = 10, dodge_rate = 10):
        super().__init__(master, hp, attack, defense, dodge_rate)

    def begin(self, opponent: Pokemon):
        super().begin(self)

    def initialize_skills(self):
        return [skills.Blank(damage = self.attack)]

    def type_effectiveness(self, opponent: Pokemon):
        return 0