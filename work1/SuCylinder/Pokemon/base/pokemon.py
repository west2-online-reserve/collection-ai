from __future__ import annotations
from .skills import Skill
from .effects import Effect
import random
from misc.tools import printWithDelay


class Pokemon:
    name: str
    type: str
    effect_list = ["中毒", "寄生种子", "烧伤"]

    def __init__(self, hp: int, attack: int, defense: int, dodge_chance: int) -> None:
        # 初始化 Pokemon 的属性
        self.cant_move = False
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses = []
        self.dodge_chance = dodge_chance
        self.damage_reduction = 0

    def get_max_hp(self):
        return self.max_hp

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        printWithDelay(f"{self.name} 使用了 {skill.name}!")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        printWithDelay(
            f"{self.name} 恢复了 {amount} HP! 现在的 HP: {self.hp}/{self.max_hp}"
        )

    def receive_damage(self, damage, type: str):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        if type not in self.effect_list:
            damage -= self.defense
            if damage <= 0:
                printWithDelay(f"{self.name} 防御了这次攻击!")
                return
        damage = damage * (1 - self.damage_reduction)
        damage = round(damage)
        self.hp -= damage
        printWithDelay(f"{self.name} 受到了 {type} 的 {damage} 点伤害!", end=" ")
        printWithDelay(f"当前 HP: {self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.alive = False
            printWithDelay(f"{self.name} 倒下了!")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                status.effect_clear(self)
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def dodged(self):
        if random.randint(1, 100) <= self.dodge_chance:
            printWithDelay(f"{self.name} 闪避了这次攻击!")
            return True
        else:
            return False

    def begin(self):
        # 新回合开始时触发的方法
        raise NotImplemented

    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"
