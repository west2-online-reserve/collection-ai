from __future__ import annotations
import random
import effects
from Pokemon.weather import Weather
from effects import Effect

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skills import Skill


class Pokemon:
    type: str

    def __init__(self, name: str, hp: int, attack: int, defense: int, sp_attack: int, sp_defense: int, dodge: int,
                 speed: int, ) -> None:
        # 初始化 Pokemon 的属性
        self.base_name = name
        self.tag = ""
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.sp_attack = sp_attack
        self.sp_defense = sp_defense
        self.dodge = dodge
        self.speed = speed
        self.alive = True
        self.skills = []
        self.statuses = []
        self.fire_turn = 0  # 给火属性加的，单独写到火属性类里面好麻烦
        self.base_attack = attack
        self.base_sp_attack = sp_attack
        self.base_defense = defense
        self.base_sp_defense = sp_defense

    @property
    def name(self) -> str:
        return f"{self.tag} {self.base_name}"

    @name.setter
    def name(self, value: str) -> None:
        self.base_name = value

    def reset(self):
        self.fire_turn = 0
        self.attack = self.base_attack
        self.sp_attack = self.base_sp_attack
        self.defense = self.base_defense
        self.sp_defense = self.base_sp_defense

    def use_skill(self, skill: Skill, opponent: Pokemon, weather: Weather):
        # 检查是否被麻痹
        for status in self.statuses:
            if isinstance(status, effects.ParalysisEffect):
                print(f"{self.name} 被麻痹了，本回合无法行动！")
                self.statuses.remove(status)
                return None
        # 使用技能
        print(f"{self.name} 使用了 {skill.name}！")
        return skill.execute(self, opponent, weather)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 回复了 {amount} HP! 剩余 HP: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage, damage_type: str, **kwargs):
        # 检查是否闪避
        if random.randint(1, 100) <= self.dodge:
            print(f"{self.name} 闪避了本次攻击！")
            if isinstance(self, ElectricPokemon):
                print("【电属性特性】 闪避成功时，立即发动一次技能！")
                return "electric_dodge"
            return "dodge"
        # 检查是否有护盾效果
        is_shield = False
        for status in self.statuses[:]:
            if isinstance(status, effects.ShieldEffect):
                is_shield = True
                # 生效一次
                self.statuses.remove(status)
                break

        if is_shield:
            damage = int(damage * 0.5)
            print(f'护盾挡下了一半攻击，伤害减半为 {damage} 点')

        # 根据攻击类型选择对应防御类型来减免伤害
        if damage_type == "physical":
            damage_reduction = self.defense
        else:
            damage_reduction = self.sp_defense
        damage -= damage_reduction
        # 检查damage是否整数
        if not isinstance(damage, int):
            damage = int(damage)

        if damage <= 0:
            print(f"{self.name} 的防御完美抵挡了本次攻击！")
            return None

        self.hp -= damage
        print(
            f"{self.name} 受到了 {damage} 伤害! 剩余HP: {max(self.hp, 0)}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 昏厥了!")

        return None

    def add_status_effect(self, effect: Effect) -> None:
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            if isinstance(status, effects.ParalysisEffect):
                continue

            status.apply(self)
            status.decrease_duration()
            if status.duration is not None and status.duration <= 0:
                print(f"{self.name} 的 {status.name} 效果结束了！")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self, weather):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name}({self.type})"


class GrassPokemon(Pokemon):
    type = "草属性"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        if opponent.type == "Water":
            return 2.0
        elif opponent.type == "Fire":
            return 0.5
        return 1.0

    def begin(self, weather):
        # 每个回合开始时执行草属性特性
        # 草属性特性：每回合恢复最大 HP 的 10%
        if weather.current_weather == "Sunny":
            heal_rate = 0.14
        else:
            heal_rate = 0.1
        amount = int(self.max_hp * heal_rate)
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"【草属性特性】 {self.name} 在回合开始时回复 {amount} HP! 当前HP: {self.hp}/{self.max_hp}")


class WaterPokemon(Pokemon):
    type = "水属性"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        if opponent.type == "Fire":
            return 2.0
        elif opponent.type == "Grass":
            return 0.5
        return 1.0

    def begin(self, weather):
        # 水属性特性：受到伤害时，有 50% 的几率减免 30% 的伤害
        if weather.current_weather == 'Rain':
            redu_chance = 70
            redu_damage = 0.35
        elif weather.current_weather == "Sunny":
            redu_chance = 30
            redu_damage = 0.25
        else:
            redu_chance = 50
            redu_damage = 0.3

        return redu_chance, redu_damage


class FirePokemon(Pokemon):
    type = "火属性"

    def type_effectiveness(self, opponent: Pokemon):
        if opponent.type == "Grass":
            return 2.0
        if opponent.type == "Water":
            return 0.5
        return 1.0

    def begin(self, weather):
        pass

    def fire_atk_increase(self):
    # 每次造成伤害，叠加 10% 攻击力，最多 4 层
        if self.fire_turn < 4:
            self.fire_turn += 1
            self.attack = int(self.base_attack*(1+self.fire_turn*0.1))
            self.sp_attack = int(self.base_sp_attack * (1 + self.fire_turn * 0.1))
            print(f"【火属性特性】 {self.name} 攻击力叠加 {self.fire_turn} 层，造成攻击力 {int((self.fire_turn*0.1+1)*100)}% 伤害！")
        else:
            print(f"【火属性特性】 {self.name} 攻击力已叠加至最大 4 层，造成攻击力 140% 伤害！")


class ElectricPokemon(Pokemon):
    type = "电属性"

    def type_effectiveness(self, opponent: Pokemon):
        if opponent.type == "Water":
            return 2.0
        if opponent.type == "Grass":
            return 0.5
        return 1.0

    def begin(self, weather):
        if weather.current_weather == "Thunderstorm":
            self.dodge = 30
        else:
            self.dodge = 20