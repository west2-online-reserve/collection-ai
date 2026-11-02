from __future__ import annotations

import random
from typing import List, Optional

import skills
from effects import Effect
from skills import Skill


class Pokemon:
    """所有宝可梦的基类。"""

    name: str = "Pokemon"
    type: str = "Normal"

    def __init__(self, hp: int, attack: int, defense: int, evasion_rate: float = 0.0):
        # 基础属性
        self.hp: int = hp
        self.max_hp: int = hp
        self.attack: int = attack
        self.defense: int = defense
        self.evasion_rate: float = evasion_rate  # 闪避率(0~1)
        # 运行时状态
        self.skills: List[Skill] = self.initialize_skills()
        self.alive: bool = True
        self.statuses: List[Effect] = []
        self.attack_stacks: int = 0  # 火属性叠加层数(0~4)
        self.shield_reduction: float = 0.0  # 护盾减伤比例
        self.evasion_bonus: float = 0.0  # 对手对我施加的额外闪避(蓄能爆炎)
        self.paralysis_skip: bool = False  # 麻痹是否令本回合失手

    # ---- 抽象接口 ----
    def initialize_skills(self) -> List[Skill]:
        raise NotImplementedError

    def type_effectiveness(self, opponent: "Pokemon") -> float:
        """针对敌方属性计算伤害倍率，子类实现。"""
        raise NotImplementedError

    # ---- 通用能力 ----
    def effective_attack(self) -> int:
        """含叠加增益的实际攻击力（火属性每层 +10%，最多 4 层）。"""
        return int(self.attack * (1 + 0.1 * self.attack_stacks))

    def use_skill(self, skill: Skill, opponent: "Pokemon"):
        """使用技能：先结算麻痹/闪避，命中后再执行技能效果。"""
        if not self.alive:
            return
        if self.paralysis_skip:
            self.paralysis_skip = False
            print(f"{self.name} is paralyzed and cannot move!")
            return
        print(f"{self.name} uses {skill.name}")
        # 闪避判定：攻击型技能会被对手闪避
        evade_chance = min(opponent.evasion_rate + opponent.evasion_bonus, 0.95)
        if skill.is_attack and opponent.alive and random.random() < evade_chance:
            print(f"{opponent.name} dodged {skill.name}!")
            # 电属性被动：成功躲闪后可立即反击一次（自动随机选招，避免阻塞）
            if opponent.type == "Thunder" and opponent.alive:
                print(f"{opponent.name}'s Thunder passive triggers a counter attack!")
                counter = random.choice(opponent.skills)
                opponent.use_skill(counter, self)
            return
        skill.execute(self, opponent)
        # 火属性被动：造成攻击伤害后叠加攻击力（最多 4 层）
        if self.type == "Fire" and skill.is_attack and opponent.hp < opponent.max_hp:
            if self.attack_stacks < 4:
                self.attack_stacks += 1
                print(f"{self.name}'s attack rises ({self.attack_stacks}/4 stacks)!")

    def heal_self(self, amount: int):
        if not isinstance(amount, int):
            amount = int(amount)
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    def modify_incoming_damage(self, damage: int) -> int:
        """受到攻击时对伤害的额外修正（护盾/水属性减伤），子类可覆盖。"""
        # 护盾：消耗一次减伤
        if self.shield_reduction > 0:
            damage = int(damage * (1 - self.shield_reduction))
            self.shield_reduction = 0.0
            print(f"{self.name}'s shield reduces the incoming damage!")
        return damage

    def receive_damage(self, damage: int):
        """计算伤害并减去防御力，更新 HP。"""
        if not isinstance(damage, int):
            damage = int(damage)
        damage = self.modify_incoming_damage(damage)
        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return
        self.hp -= damage
        print(f"{self.name} received {damage} damage! Remaining HP: {self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")

    def add_status_effect(self, effect: Effect):
        self.statuses.append(effect)

    def apply_status_effect(self):
        """应用所有状态效果并移除到期效果。"""
        for status in self.statuses[:]:
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)

    def begin(self):
        """新回合开始钩子：重置临时闪避、结算状态效果、执行属性被动。"""
        self.evasion_bonus = 0.0
        self.apply_status_effect()

    def __str__(self) -> str:
        return f"{self.name} type: {self.type} (HP: {self.hp}/{self.max_hp})"


# ------------------------- 属性类 -------------------------
class GrassPokemon(Pokemon):
    """草属性：克制水、被火克制；每回合回复 10% 最大 HP。"""

    type = "Grass"

    def type_effectiveness(self, opponent: "Pokemon") -> float:
        effectiveness = 1.0
        if opponent.type == "Water":
            effectiveness = 2.0
        elif opponent.type == "Fire":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        super().begin()
        self.grass_attribute()

    def grass_attribute(self):
        amount: int = int(self.max_hp * 0.1)
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP (Grass passive)! Current HP: {self.hp}/{self.max_hp}")


class ThunderPokemon(Pokemon):
    """电属性：克制水、被草克制；被动"成功躲闪后立即使用一次技能"。

    躲闪反击逻辑集中在 Pokemon.use_skill 中处理（命中前闪避判定处），
    因此 begin 无需额外行为，只重置临时状态即可。
    """

    type = "Thunder"

    def type_effectiveness(self, opponent: "Pokemon") -> float:
        effectiveness = 1.0
        if opponent.type == "Water":
            effectiveness = 2.0
        elif opponent.type == "Grass":
            effectiveness = 0.5
        return effectiveness


class WaterPokemon(Pokemon):
    """水属性：克制火、被电克制；受到伤害时 50% 几率减免 30%。"""

    type = "Water"

    def type_effectiveness(self, opponent: "Pokemon") -> float:
        effectiveness = 1.0
        if opponent.type == "Fire":
            effectiveness = 2.0
        elif opponent.type == "Thunder":
            effectiveness = 0.5
        return effectiveness

    def modify_incoming_damage(self, damage: int) -> int:
        damage = super().modify_incoming_damage(damage)
        # 水属性被动：50% 几率减免 30% 伤害
        if damage > 0 and random.random() < 0.5:
            damage = int(damage * 0.7)
            print(f"{self.name}'s Water passive reduces the damage!")
        return damage


class FirePokemon(Pokemon):
    """火属性：克制草、被水克制；每次造成伤害叠加 10% 攻击（最多 4 层）。

    叠加逻辑集中在 Pokemon.use_skill（造成攻击伤害后）处理。
    """

    type = "Fire"

    def type_effectiveness(self, opponent: "Pokemon") -> float:
        effectiveness = 1.0
        if opponent.type == "Grass":
            effectiveness = 2.0
        elif opponent.type == "Water":
            effectiveness = 0.5
        return effectiveness


# ------------------------- 具体宝可梦 -------------------------
class Bulbasaur(GrassPokemon):
    """妙蛙种子：HP100 攻35 防10 躲闪10%。"""

    name = "Bulbasaur"

    def __init__(self, hp: int = 100, attack: int = 35, defense: int = 10, evasion_rate: float = 0.10):
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self) -> List[Skill]:
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class PikaChu(ThunderPokemon):
    """皮卡丘：HP80 攻35 防5 躲闪30%。"""

    name = "PikaChu"

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 5, evasion_rate: float = 0.30):
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self) -> List[Skill]:
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Squirtle(WaterPokemon):
    """杰尼龟：HP80 攻25 防20 躲闪20%。"""

    name = "Squirtle"

    def __init__(self, hp: int = 80, attack: int = 25, defense: int = 20, evasion_rate: float = 0.20):
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self) -> List[Skill]:
        return [skills.AquaJet(), skills.Shield()]


class Charmander(FirePokemon):
    """小火龙：HP80 攻35 防15 躲闪10%。"""

    name = "Charmander"

    def __init__(self, hp: int = 80, attack: int = 35, defense: int = 15, evasion_rate: float = 0.10):
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self) -> List[Skill]:
        return [skills.Ember(), skills.FlameCharge()]


class Vulpix(FirePokemon):
    """六尾：HP70 攻40 防8 躲闪15%。

    技能：火花 + 嚎叫（主动叠加攻击力，配合火属性被动更快叠满 4 层）。
    """

    name = "Vulpix"

    def __init__(self, hp: int = 70, attack: int = 40, defense: int = 8, evasion_rate: float = 0.15):
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self) -> List[Skill]:
        return [skills.Ember(burn_chance=20), skills.Howl()]
