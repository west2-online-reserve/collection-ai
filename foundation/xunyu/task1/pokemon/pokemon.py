# pokemon.py - 宝可梦类定义

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from effects import Effect

import random
import skills
from skills import Skill


class Pokemon:
    """宝可梦基类"""
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        self.hp = hp  
        self.max_hp = hp  
        self.attack = attack  
        self.defense = defense  
        self.skills = self.initialize_skills()  # 技能列表
        self.alive = True  # 是否存活
        self.statuses = []  # 状态效果列表
        self.evasion = 0.0  # 闪避率
        self.paralyzed = False  # 麻痹
        self.confused_turns = 0  # 混乱剩余回合
        self.leech_seed_turns = 0  # 寄生种子剩余回合
        self.leech_seed_source = None  
        self.shield = False  # 护盾
        self.charging = False  # 蓄力中
        self.charged_skill = None  # 蓄力的技能
        self.attack_boost = 0  # 火属性攻击叠加层数

    def initialize_skills(self):
        # 子类实现具体技能
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon) -> None:
        # 使用技能攻击对手
        print(f"{self.name} 使用了 {skill.name}！")

        if not self.can_act():  # 检查麻痹/混乱
            return

        if opponent.check_dodge(self):  # 检查对方闪避
            return

        skill.execute(self, opponent)

    def can_act(self) -> bool:
        # 检查是否可行动，返回True表示可以
        if self.paralyzed and random.random() < 0.25:
            print(f"{self.name} 麻痹了，无法行动！")
            return False

        if self.confused_turns > 0:
            self.confused_turns -= 1
            if random.random() < 0.3:
                dmg = max(int(self.attack * 0.3), 1)
                self.hp -= dmg
                print(f"{self.name} 混乱了，打了自己 {dmg} 点伤害！")
                if self.hp <= 0:
                    self.alive = False
                    print(f"{self.name} 倒下了！")
                return False
            if self.confused_turns <= 0:
                print(f"{self.name} 恢复了清醒！")
        return True

    def check_dodge(self, attacker: Pokemon) -> bool:
        # 检查是否能躲闪，返回True表示躲闪成功
        if random.random() < self.evasion:
            print(f"{attacker.name} 的攻击被躲开了！{self.name} 灵巧地闪避了！")
            if self.type == "Electric" and self.alive:
                dmg_skills = [s for s in self.skills if s.multiplier > 0]
                if dmg_skills:
                    cs = random.choice(dmg_skills)
                    print(f"[电属性] {self.name} 发动反击！")
                    cs.execute(self, attacker)  # 直接执行，不触发再次闪避
            return True
        return False

    def receive_damage(self, damage: int) -> None:
        # 受到伤害：防御减免 + 水属性被动 + 护盾
        damage = max(damage - self.defense, 1)

        if self.type == "Water" and random.random() < 0.5:
            damage = int(damage * 0.7)  # 水属性减伤30%
            print(f"[水属性] {self.name} 的水之柔体减免了伤害！")

        if self.shield:
            damage = int(damage * 0.5)  # 护盾减半
            self.shield = False
            print(f"{self.name} 的护盾吸收了部分伤害！")

        self.hp -= damage
        print(f"{self.name} 受到 {damage} 点伤害！剩余HP：{self.hp}/{self.max_hp}")

        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 倒下了！")

    def heal_self(self, amount: int) -> None:

        self.hp = min(self.hp + amount, self.max_hp)

    def add_status_effect(self, effect: Effect) -> None:
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        # 应用所有状态效果
        for s in self.statuses[:]:#遍历副本 避免修改列表使得索引出问题 错误点之一
            s.apply(self)
            s.decrease_duration()
            if s.duration <= 0:
                print(f"{self.name} 的 {s.name} 效果解除了。")
                self.statuses.remove(s)

    def begin(self) -> None:
        # 回合开始：处理状态和寄生种子
        self.apply_status_effect()
        if not self.alive:
            return

        if self.leech_seed_turns > 0 and self.leech_seed_source and self.leech_seed_source.alive:
            dmg = max(int(self.max_hp * 0.1), 1)
            self.hp -= dmg
            self.leech_seed_source.heal_self(dmg)
            print(f"[寄生种子] {self.name} 被吸取了 {dmg} 点HP！")
            self.leech_seed_turns -= 1
            if self.leech_seed_turns <= 0:
                self.leech_seed_source = None
            if self.hp <= 0:
                self.alive = False
                print(f"{self.name} 倒下了！")

    def type_effectiveness(self, opponent: Pokemon) -> float:
        # 由子类实现属性克制
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}（{self.type}属性）HP:{self.hp}/{self.max_hp}"


class GrassPokemon(Pokemon):
    # 草属性：每回合回复10%HP
    type = "Grass"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Water":
            effectiveness = 2.0
        elif opponent.type == "Fire":
            effectiveness = 0.5
        return effectiveness

    def begin(self) -> None:
        super().begin()
        if not self.alive:
            return
        heal = max(int(self.max_hp * 0.1), 1)
        if self.hp < self.max_hp:
            self.heal_self(heal)
            print(f"[草属性] {self.name} 进行光合作用，回复了 {heal} 点HP！({self.hp}/{self.max_hp})")


class FirePokemon(Pokemon):
    # 火属性：每次攻击叠加10%攻击力（在skills中实现）
    type = "Fire"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Grass":
            effectiveness = 2.0
        elif opponent.type == "Water":
            effectiveness = 0.5
        return effectiveness


class WaterPokemon(Pokemon):
    # 水属性：受到伤害50%减免30%（在receive_damage中实现）
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Fire":
            effectiveness = 2.0
        elif opponent.type == "Electric":
            effectiveness = 0.5
        return effectiveness


class ElectricPokemon(Pokemon):
    # 电属性：闪避时反击（在check_dodge中实现）
    type = "Electric"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Water":
            effectiveness = 2.0
        elif opponent.type == "Psychic":
            effectiveness = 0.5
        return effectiveness


class PsychicPokemon(Pokemon):
    # 超能属性：30%概率每回合回复15%HP
    type = "Psychic"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Electric":
            effectiveness = 2.0
        elif opponent.type == "Grass":
            effectiveness = 0.5
        return effectiveness

    def begin(self) -> None:
        super().begin()
        if not self.alive:
            return
        if random.random() < 0.3:
            heal = max(int(self.max_hp * 0.15), 1)
            self.heal_self(heal)
            print(f"[超能属性] {self.name} 的超能力回复了 {heal} 点HP！({self.hp}/{self.max_hp})")


class Pikachu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 5) -> None:
        super().__init__(hp, attack, defense)
        self.evasion = 0.30

    def initialize_skills(self) -> list:
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"

    def __init__(self, hp: int = 100, attack: int = 35, defense: int = 10) -> None:
        super().__init__(hp, attack, defense)
        self.evasion = 0.10

    def initialize_skills(self) -> list:
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, hp: int = 80, attack: int = 25, defense: int = 20) -> None:
        super().__init__(hp, attack, defense)
        self.evasion = 0.20

    def initialize_skills(self) -> list:
        return [skills.AquaJet(), skills.Shield()]


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 15) -> None:
        super().__init__(hp, attack, defense)
        self.evasion = 0.10

    def initialize_skills(self) -> list:
        return [skills.Ember(), skills.FlameCharge()]


class Mew(PsychicPokemon):
    name = "梦幻"

    def __init__(self, hp: int = 90, attack: int = 40, defense: int = 10) -> None:
        super().__init__(hp, attack, defense)
        self.evasion = 0.20

    def initialize_skills(self) -> list:
        return [skills.PsyShock(), skills.Recover()]


ALL_POKEMON = [Pikachu, Bulbasaur, Squirtle, Charmander, Mew]  # 所有宝可梦
