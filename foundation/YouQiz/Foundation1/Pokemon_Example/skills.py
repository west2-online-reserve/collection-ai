from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


def deal_damage(
    user: Pokemon,
    opponent: Pokemon,
    power: float,  # 伤害倍率
    bonus_dodge: float = 0.0,   # 额外提供给对手的闪避率
) -> bool:
    """普通攻击：闪避 → 属性 → 防御 → 水系/护盾减伤。"""
    if random.random() < min(0.95, opponent.dodge + bonus_dodge):
        print(f"{opponent.name} 闪避了攻击！")
        # 电属性被动：成功闪避后立刻反击
        if opponent.type == "电":
            counter = random.choice(opponent.skills)
            print(f"{opponent.name} 触发了雷电反射！")
            counter.execute(opponent, user)
        return False

    # 属性克制倍率 × 攻击力 × 招式倍率
    damage = int(user.attack * power * user.type_effectiveness(opponent))

    # 水属性被动：50% 概率减伤 30%
    if opponent.type == "水" and random.random() < 0.5:
        damage = int(damage * 0.7)
        print(f"{opponent.name} 的水之守护发动，伤害降低 30%！")

    # 护盾：受到的伤害减半
    if any(status.name == "护盾" for status in opponent.statuses):
        damage = int(damage * 0.5)

    if user.type_effectiveness(opponent) > 1.0:
        print("效果拔群！")
    elif user.type_effectiveness(opponent) < 1.0:
        print("效果不佳……")

    before = opponent.hp
    opponent.receive_damage(damage)

    # 火属性被动：造成实际伤害后攻击力 +10%，最多提高到基础的 1.4 倍
    if opponent.hp < before and user.type == "火":
        cap = int(user.base_attack * 1.4)
        if user.attack < cap:
            user.attack = min(cap, int(user.attack * 1.1))
            print(f"{user.name} 的攻击力提升到 {user.attack}！")
    return True


class Skill(ABC):
    name: str
    requires_charge: bool = False  # 是否需要蓄力（目前只有蓄能爆炎）

    def __init__(self) -> None:
        pass

    @abstractmethod
    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class SeedBomb(Skill):
    """种子炸弹：造成草属性伤害，命中后 15% 概率使对手中毒。"""
    name = "种子炸弹"

    def __init__(self, power: float = 1.0, activation_chance: float = 0.15) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")

        if deal_damage(user, opponent, self.power):
            if random.uniform(0, 1) < self.activation_chance:
                opponent.add_status_effect(effects.PoisonEffect())
                print(f"{opponent.name} 中毒了！")
            else:
                print(f"{self.name} 没有触发中毒。")


class ParasiticSeeds(Skill):
    name = "寄生种子"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} 获得了回复效果。")
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} 中毒了！")


class Thunderbolt(Skill):
    name = "十万伏特"

    def __init__(self, power: float = 1.4, activation_chance: float = 0.10) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        if deal_damage(user, opponent, self.power):
            if random.uniform(0, 1) < self.activation_chance:
                opponent.add_status_effect(effects.ParalysisEffect())
                print(f"{opponent.name} 麻痹了！")


class ThunderPunch(Skill):
    name = "雷电拳"

    def __init__(self, power: float = 1.2, activation_chance: float = 0.20) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        if deal_damage(user, opponent, self.power):
            if random.uniform(0, 1) < self.activation_chance:
                opponent.add_status_effect(effects.ParalysisEffect())
                print(f"{opponent.name} 麻痹了！")


class QuickAttack(Skill):
    name = "电光一闪"

    def __init__(self, power: float = 1.0, activation_chance: float = 0.10) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        deal_damage(user, opponent, self.power)
        if (
            user.alive
            and opponent.alive
            and random.uniform(0, 1) < self.activation_chance
        ):
            print(f"{user.name} 再次发动攻击！")
            deal_damage(user, opponent, self.power)


class AquaJet(Skill):
    name = "水枪"

    def __init__(self, power: float = 1.4) -> None:
        super().__init__()
        self.power = power

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        deal_damage(user, opponent, self.power)


class HydroPump(Skill):
    name = "水炮"

    def __init__(self, power: float = 1.6) -> None:
        super().__init__()
        self.power = power

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        deal_damage(user, opponent, self.power)


class Shield(Skill):
    name = "护盾"

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        user.add_status_effect(effects.ShieldEffect())


class Ember(Skill):
    name = "火花"

    def __init__(self, power: float = 1.0, activation_chance: float = 0.10) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        print(f"{user.name} 使用了 {self.name}！")
        if deal_damage(user, opponent, self.power):
            if random.uniform(0, 1) < self.activation_chance:
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.name} 烧伤了！")


class FlameCharge(Skill):
    name = "蓄能爆炎"
    requires_charge = True

    def __init__(self, power: float = 3.0, activation_chance: float = 0.80) -> None:
        super().__init__()
        self.power = power
        self.activation_chance = activation_chance

    def execute(self, user: Pokemon, opponent: Pokemon) -> None:
        if user.charging_skill is not self:
            user.charging_skill = self
            print(f"{user.name} 开始蓄力 {self.name}……")
            return
        user.charging_skill = None
        print(f"{user.name} 释放了 {self.name}！")
        if deal_damage(user, opponent, self.power, bonus_dodge=0.20):
            if random.uniform(0, 1) < self.activation_chance:
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.name} 烧伤了！")