from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random

TYPE_EFFECTIVENESS: dict[str, dict[str, float]] = {
    "Grass": {"Grass": 1.0, "Fire": 0.5, "Water": 2.0, "Electric": 2.0},
    "Fire": {"Grass": 2.0, "Fire": 1.0, "Water": 0.5, "Electric": 1.0},
    "Water": {"Grass": 0.5, "Fire": 2.0, "Water": 1.0, "Electric": 0.5},
    "Electric": {"Grass": 0.5, "Fire": 1.0, "Water": 2.0, "Electric": 1.0},
}


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills: list[Skill] = self.initialize_skills()
        self.alive = True
        self.statuses: list[Effect] = []

    def initialize_skills(self) -> list[Skill]:
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name} uses {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount: int | float) -> None:
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)
        # 普通治疗只治疗存活成员，不提供复活
        if not self.alive:
            return
        if amount <= 0:
            return

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    # 新增lose_hp，统一扣血处理
    def lose_hp(self, amount: int | float) -> None:
        if not self.alive or amount <= 0:
            return
        if not isinstance(amount, int):
            amount = int(amount)
        initial_hp = self.hp
        self.hp -= amount
        if self.hp <= 0:  # 防止血量出现负数
            self.hp = 0
        print(
            f"{self.name} received {initial_hp - self.hp} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")

    def receive_damage(self, damage: int | float) -> None:
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return
        self.lose_hp(damage)

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        if not self.alive:
            return
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)
            if self.hp <= 0:  # hp小于0，结束循环
                break

    # 克制计算倍率由父类统一实现
    def type_effectiveness(self, opponent: Pokemon) -> float:
        # 根据对手的类型计算效果倍率
        return TYPE_EFFECTIVENESS.get(self.type, {}).get(opponent.type, 1.0)

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


# GrassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

    def begin(self):
        # 每个回合开始时执行草属性特性
        self.grass_attribute()

    def grass_attribute(self) -> None:
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        print("Grass attribute activates! Healing 10% of max HP.")
        self.heal_self(amount)


# Bulbasaur 类，继承自 GrassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp: int = 100, attack: int = 35, defense: int = 10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class WaterPokemon(Pokemon):
    type = "Water"

    def receive_damage(self, damage: int | float) -> None:
        if not self.alive:
            return
        # 受到伤害时，有 50% 的几率减免 30% 的伤害
        if random.randint(1, 100) <= 50:
            damage *= 0.7
            print(f"{self.name}'s Water attribute reduces the damage by 30%!")
        super().receive_damage(damage)


class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp: int = 80, attack: int = 25, defense: int = 20) -> None:
        # 初始化 Squirtle 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 AquaJet
        return [skills.AquaJet()]
