# skills.py - 技能类

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon

import random
from effects import PoisonEffect, BurnEffect


class Skill:
    """技能基类"""
    name: str

    def __init__(self, multiplier: float = 0) -> None:
        self.multiplier = multiplier  # 攻击力倍率
        self.needs_charge = False  # 是否需要蓄力

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        raise NotImplementedError

    def calculate_damage(self, user: "Pokemon", opponent: "Pokemon") -> int:
        # 攻击力 × 倍率 × 属性克制
        base = int(user.attack * self.multiplier)  # 基础伤害
        effectiveness = user.type_effectiveness(opponent)  # 属性克制
        total = int(base * effectiveness)

        if effectiveness == 2.0:
            print("效果拔群！")
        elif effectiveness == 0.5:
            print("效果不太好...")

        # 火属性被动：叠加攻击力
        if user.type == "Fire" and self.multiplier > 0:
            boost = 1 + 0.1 * user.attack_boost
            total = int(total * boost)
            user.attack_boost = min(user.attack_boost + 1, 4)
            print(f"[火属性] {user.name} 的攻击力提升了！({user.attack_boost}/4)")

        return total


class Thunderbolt(Skill):
    # 十万伏特：1.4倍电属性伤害，10%麻痹
    name = "十万伏特"

    def __init__(self) -> None:
        super().__init__(multiplier=1.4)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.1:
            opponent.paralyzed = True
            print(f"{opponent.name} 麻痹了！")


class QuickAttack(Skill):
    # 电光一闪：1.0倍攻击，10%触发第二次
    name = "电光一闪"

    def __init__(self) -> None:
        super().__init__(multiplier=1.0)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.1:
            print(f"{user.name} 的电光一闪再次攻击！")
            total2 = self.calculate_damage(user, opponent)
            opponent.receive_damage(total2)


class SeedBomb(Skill):
    # 种子炸弹：草属性伤害，15%中毒
    name = "种子炸弹"

    def __init__(self) -> None:
        super().__init__(multiplier=1.2)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.15:
            opponent.add_status_effect(PoisonEffect(duration=3))
            print(f"{opponent.name} 中毒了！")


class ParasiticSeeds(Skill):
    # 寄生种子：种下种子，每回合吸血
    name = "寄生种子"

    def __init__(self) -> None:
        super().__init__(multiplier=0)  # 不直接造成伤害

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if opponent.leech_seed_turns > 0:
            print("已经种过寄生种子了！")
            return
        opponent.leech_seed_turns = 3  # 持续3回合
        opponent.leech_seed_source = user
        print(f"{opponent.name} 被种下了寄生种子！")


class AquaJet(Skill):
    # 水枪：1.4倍水属性伤害
    name = "水枪"

    def __init__(self) -> None:
        super().__init__(multiplier=1.4)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)


class Shield(Skill):
    # 护盾：不造成伤害，下回合减伤50%
    name = "护盾"

    def __init__(self) -> None:
        super().__init__(multiplier=0)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.shield = True
        print(f"{user.name} 使用了护盾！下次伤害减半！")


class Ember(Skill):
    # 火花：1.0倍火属性伤害，10%烧伤
    name = "火花"

    def __init__(self) -> None:
        super().__init__(multiplier=1.0)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.1:
            opponent.add_status_effect(BurnEffect(damage=10, duration=2))
            print(f"{opponent.name} 烧伤了！")


class FlameCharge(Skill):
    # 蓄能爆炎：3.0倍火属性伤害，需蓄力，80%烧伤
    # 面对该技能时敌方闪避率增加20%
    name = "蓄能爆炎"

    def __init__(self) -> None:
        super().__init__(multiplier=3.0)
        self.needs_charge = True  # 需要蓄力一回合
        self.raise_enemy_evasion = True  # 敌方闪避率增加

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.8:
            opponent.add_status_effect(BurnEffect(damage=10, duration=2))
            print(f"{opponent.name} 被严重烧伤了！")


class PsyShock(Skill):
    # 精神冲击：1.3倍超能伤害，20%混乱
    name = "精神冲击"

    def __init__(self) -> None:
        super().__init__(multiplier=1.3)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total = self.calculate_damage(user, opponent)
        opponent.receive_damage(total)
        if random.random() < 0.2 and opponent.confused_turns <= 0:
            opponent.confused_turns = 2  # 混乱2回合
            print(f"{opponent.name} 混乱了！")


class Recover(Skill):
    # 自我再生：回复30%HP
    name = "自我再生"

    def __init__(self) -> None:
        super().__init__(multiplier=0)

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        heal = int(user.max_hp * 0.3)
        user.heal_self(heal)
        print(f"{user.name} 回复了 {heal} 点HP！({user.hp}/{user.max_hp})")
