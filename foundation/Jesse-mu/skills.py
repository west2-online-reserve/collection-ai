"""宝可梦技能（Skills）模块。

每个技能继承自 Skill 抽象基类并实现 execute。技能内部只负责"命中之后"的效果，
闪避判定等公共逻辑统一在 Pokemon.use_skill 中处理。
伤害计算公式：user.effective_attack() * multiplier * user.type_effectiveness(opponent)
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import effects
from typing import Optional

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:

    name: str = "Skill"
    # 该技能是否为"攻击型"（会造成直接伤害），供火属性被动等使用
    is_attack = False

    def __init__(self):
        pass

    def _calc_damage(self, user: "Pokemon", opponent: "Pokemon", multiplier: float):
        """按攻击力、倍率与属性克制计算最终伤害。"""
        effectiveness = user.type_effectiveness(opponent)
        raw = user.effective_attack() * multiplier * effectiveness
        return int(raw)

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    """种子炸弹：造成草属性伤害，15% 几率使对手中毒。"""

    name = "Seed Bomb"
    is_attack = True

    def __init__(self, multiplier: float = 1.0, activation_chance: int = 15):
        super().__init__()
        self.multiplier: float = multiplier
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} used {self.name}, dealing {damage} damage to {opponent.name}"
        )
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} is poisoned by {self.name}!")


class ParasiticSeeds(Skill):
    """寄生种子：每回合吸取对手 10% 最大 HP 并回复自身，持续 3 回合。"""

    name = "Parasitic Seeds"

    def __init__(self, percentage: float = 0.1, duration: int = 3):
        super().__init__()
        self.percentage: float = percentage
        self.duration: int = duration

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        drain = effects.DrainEffect(self.percentage, self.duration)
        drain.target = opponent
        user.add_status_effect(drain)
        print(f"{user.name} plants {self.name} on {opponent.name}!")


class Thunderbolt(Skill):
    """十万伏特：造成 1.4 倍电属性伤害，10% 几率麻痹对手。"""

    name = "Thunderbolt"
    is_attack = True

    def __init__(self, multiplier: float = 1.4, paralyze_chance: int = 10):
        super().__init__()
        self.multiplier = multiplier
        self.paralyze_chance = paralyze_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} used {self.name}, dealing {damage} damage to {opponent.name}"
        )
        if random.randint(1, 100) <= self.paralyze_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name} is paralyzed by {self.name}!")


class QuickAttack(Skill):
    """电光一闪：造成 1.0 倍伤害，10% 几率触发第二次攻击。"""

    name = "Quick Attack"
    is_attack = True

    def __init__(self, multiplier: float = 1.0, second_chance: int = 10):
        super().__init__()
        self.multiplier: float = multiplier
        self.second_chance: int = second_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} used {self.name}, dealing {damage} damage to {opponent.name}"
        )
        if opponent.alive and random.randint(1, 100) <= self.second_chance:
            print(f"{user.name} attacks a second time!")
            damage2: int = self._calc_damage(user, opponent, self.multiplier)
            opponent.receive_damage(damage2)


class AquaJet(Skill):
    """水枪：造成 140% 水属性伤害。"""

    name = "Aqua Jet"
    is_attack = True

    def __init__(self, multiplier: float = 1.4):
        super().__init__()
        self.multiplier: float = multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} used {self.name}, dealing {damage} damage to {opponent.name}"
        )


class Shield(Skill):
    """护盾：使自身下一回合受到的伤害减少 50%。"""

    name = "Shield"

    def __init__(self, reduction: float = 0.5):
        super().__init__()
        self.reduction: float = reduction

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        user.shield_reduction = self.reduction
        print(f"{user.name} raises a {self.name}! Incoming damage reduced.")


class Ember(Skill):
    """火花：造成 100% 火属性伤害，10% 几率使目标烧伤。"""

    name = "Ember"
    is_attack = True

    def __init__(self, multiplier: float = 1.0, burn_chance: int = 10):
        super().__init__()
        self.multiplier: float = multiplier
        self.burn_chance: int = burn_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} used {self.name}, dealing {damage} damage to {opponent.name}"
        )
        if random.randint(1, 100) <= self.burn_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} is burned by {self.name}!")


class FlameCharge(Skill):
    """蓄能爆炎：需 1 回合蓄力，蓄力时敌方闪避率 +20%，
    释放时造成 300% 火属性伤害并有 80% 几率烧伤。"""

    name = "Flame Charge"
    is_attack = True

    def __init__(
        self,
        multiplier: float = 3.0,
        burn_chance: int = 80,
        evade_bonus: float = 0.2,
    ):
        super().__init__()
        self.multiplier: float = multiplier
        self.burn_chance: int = burn_chance
        self.evade_bonus: float = evade_bonus
        self.charging = False  # 蓄力状态标记

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        if not self.charging:
            # 第一次使用：进入蓄力，提高对手闪避率，本回合不造成伤害
            self.charging = True
            opponent.evasion_bonus += self.evade_bonus
            print(f"{user.name} is charging {self.name}! {opponent.name} evasiveness rises.")
            return
        # 蓄力完成：释放
        self.charging = False
        damage: int = self._calc_damage(user, opponent, self.multiplier)
        opponent.receive_damage(damage)
        print(
            f"{user.name} released {self.name}, dealing {damage} damage to {opponent.name}"
        )
        if random.randint(1, 100) <= self.burn_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} is severely burned by {self.name}!")


class Howl(Skill):
    """嚎叫：提升自身攻击力叠加层数（配合火属性被动，最多 4 层）。"""

    name = "Howl"

    def __init__(self):
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        if user.attack_stacks < 4:
            user.attack_stacks += 1
            print(
                f"{user.name} howls! Attack power increased "
                f"({user.attack_stacks}/4 stacks)."
            )
        else:
            print(f"{user.name}'s attack is already at maximum!")
