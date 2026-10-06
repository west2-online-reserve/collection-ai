from __future__ import annotations
import skills
from skills import Skill
import effects
from effects import Effect, HitEffect, AttbuffEffect, DRbuffEffect, DefbuffEffect
from typing import List
import random


def printf(web):
    print(web)
    # time.sleep(0.3)
    input()


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int, avoid: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.base_attack = attack
        self.base_defense = defense
        self.attack = attack
        self.defense = defense
        self.avoid = avoid
        self.skills: List = self.initialize_skills()
        self.alive = True
        self.statuses = []
        self.DR = 0

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        if [eff for eff in self.statuses if isinstance(eff, type(effects.paralysis()))]:
            print(f"{self.name} is paralyzed")
            pass
        else:
            printf(f"{self.name} uses {skill.name}")
            skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage, fromskill: bool = False, eff: int = 1) -> int:
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            printf(f"{self.name}'s defense absorbed the attack!")
            return
        if fromskill:
            damage *= eff
            damage *= 1 - self.DR * 0.01
        self.hp -= damage
        printf(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp:.1f}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
        return damage

    def hp_lost(self, damage) -> int:
        if not isinstance(damage, int):
            damage = int(damage)

        self.hp -= damage
        printf(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp:.1f}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
        return damage

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        for existing_effect in self.statuses:
            if isinstance(existing_effect, type(effect)):
                if existing_effect.max_number > existing_effect.number:
                    existing_effect.number += 1
                    print(
                        f"{existing_effect.name}'s stacks is increased to {existing_effect.number} "
                    )
                if not (effect.noduration):
                    effect.reload()
                return
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        # 重置属性

        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            if isinstance(status, HitEffect):
                status.apply(self)
            elif isinstance(status, AttbuffEffect):
                self.attack += self.base_attack * status.amount * status.number * 0.01
            elif isinstance(status, DefbuffEffect):
                self.defense += self.base_defense * status.amount * status.number * 0.01
            elif isinstance(status, DRbuffEffect):
                self.DR += status.amount * status.number
            status.decrease_duration()
            for status in self.statuses[:]:
                print(f"DEBUG: 当前状态类型是 {type(status)}")  # 加这行！
            # ... 后面的代码不变
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        self.attack = self.base_attack
        self.defense = self.base_defense
        self.DR = 0
        self.apply_status_effect()
        printf(
            f"{self.name}'s amount:\n HP:{self.hp:.1f}/{self.max_hp}\n attack:{self.attack:.1f}\n defence: {self.defense:.1f}\n Damade reduction:{self.DR:.1f} "
        )
        # 新回合开始时触发的方法

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"

    # def att_raise(self, amount: int) -> int:
    #     self.attack = self.base_attack * (1 + amount * 0.01)

    # def def_raise(self, amount: int) -> int:
    #     self.defence = self.base_defense * (1 + amount * 0.01)


# GlassPokemon 类
class GlassPokemon(Pokemon):
    type = "Glass"

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
        super().begin()
        # 每个回合开始时执行玻璃属性特性
        self.glass_attribute()

    def glass_attribute(self):
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
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Glass":
            effectiveness = 2.0
        elif opponent_type == "Water":
            effectiveness = 0.5
        return effectiveness

    def use_skill(self, skill, opponent):
        # 执行Fire特性
        super().use_skill(skill, opponent)
        self.add_status_effect(effects.attraise())


class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Fire":
            effectiveness = 2.0
        elif opponent_type == "Glass" or opponent_type == "Electric":
            effectiveness = 0.5
        return effectiveness

    def receive_damage(self, damage, fromskill=False, eff=1):
        # 执行Water特性
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            printf(f"{self.name}'s defense absorbed the attack!")
            return
        if fromskill:
            damage *= eff
            damage *= 1 - self.DR * 0.01
        if random.randint(1, 2) == 1:
            damage *= 0.7
            print("Water-type Ability activates, reducing damage taken by 30%.")
        self.hp -= damage
        printf(
            f"{self.name} received {damage:.1f} damage! Remaining HP: {self.hp:.1f}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
        return damage


class ElectricPokemon(Pokemon):
    type = "Electric"
    avoid = False

    # 用于执行特性
    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Water":
            effectiveness = 2.0
        elif opponent_type == "Glass":
            effectiveness = 0.5
        return effectiveness


class RockPokemon(Pokemon):
    type = "Rock"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        return effectiveness

    def begin(self):
        up = (1 - self.hp / self.max_hp) / 2
        self.attack = self.base_attack * (1 + up)
        self.defense = self.base_defense * (1 + up)
        self.DR = up * 100
        self.apply_status_effect()
        printf(
            f"{self.name}'s amount:\n HP:{self.hp:.1f}/{self.max_hp}\n attack:{self.attack:.1f}\n defence: {self.defense:.1f}\n Damade reduction:{self.DR:.1f} "
        )


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GlassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=25, defense=10, avoid=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense, avoid)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(), skills.ParasiticSeeds(amount=10)]


class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(self, hp=80, attack=35, defense=15, avoid=10) -> None:
        super().__init__(hp, attack, defense, avoid)

    def initialize_skills(self):
        return [skills.Ember(), skills.FlameChargePrepare()]


class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=85, attack=25, defense=25, avoid=10):
        super().__init__(hp, attack, defense, avoid)

    def initialize_skills(self):
        return [skills.Shield(), skills.AquaJet()]


class PikaChu(ElectricPokemon):
    name = "PikaChu"

    def __init__(self, hp=80, attack=35, defense=5, avoid=20):
        super().__init__(hp, attack, defense, avoid)

    def initialize_skills(self):
        return [skills.QuickAttack(), skills.Thunderbolt()]


class Geodude(RockPokemon):
    name = "Geodude"

    def __init__(self, hp=80, attack=20, defense=20, avoid=0):
        super().__init__(hp, attack, defense, avoid)

    def initialize_skills(self):
        return [skills.RageBuild(), skills.RockFist()]
