from __future__ import annotations
import skills
import effects
from skills import Skill
from effects import Effect
import random
from typing import Literal

TYPE_EFFECTIVENESS: dict[str, dict[str, float]] = {
    "Grass": {"Grass": 1.0, "Fire": 0.5, "Water": 2.0, "Electric": 2.0, "Normal": 1.0},
    "Fire": {"Grass": 2.0, "Fire": 1.0, "Water": 0.5, "Electric": 1.0, "Normal": 1.0},
    "Water": {"Grass": 0.5, "Fire": 2.0, "Water": 1.0, "Electric": 0.5, "Normal": 1.0},
    "Electric": {
        "Grass": 0.5,
        "Fire": 1.0,
        "Water": 2.0,
        "Electric": 1.0,
        "Normal": 1.0,
    },
    "Normal": {"Grass": 1.0, "Fire": 1.0, "Water": 1.0, "Electric": 1.0, "Normal": 1.0},
}


class Pokemon:
    name: str
    type: str

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_chance: int = 0,
        charging_skill: Skill | None = None,
        extra_action_pending: bool = False,
        speed: int = 50,
    ) -> None:  # 新增dodge_chance闪避概率，charging_skill正在蓄力的技能对象
        # 初始化 Pokemon 的属性
        self.hp: int = hp
        self.max_hp: int = hp
        self.attack: int = attack
        self.defense: int = defense
        self.dodge_chance: int = dodge_chance
        self.speed: int = speed
        self.side: Literal["player", "computer"] | None = None
        self.charging_skill: Skill | None = charging_skill
        self.extra_action_pending: bool = extra_action_pending
        self.skills: list[Skill] = self.initialize_skills()
        self.alive: bool = True
        self.statuses: list[Effect] = []

    @property  # @property 装饰器将 display_name 方法变为属性访问
    def display_name(self) -> str:
        if self.side is None:
            return self.name
        return f"[{self.side}] {self.name}"

    def initialize_skills(self) -> list[Skill]:
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon) -> None:
        if self.charging_skill is not None:
            if self.charging_skill != skill:
                self.charging_skill = None  # 如果使用了不同的技能，取消蓄力状态
        skill.execute(self, opponent)

    def on_switch_out(self) -> None:
        # 主动或被迫离场时保留 HP，清除自身所有战斗效果。
        self.statuses.clear()
        self.charging_skill = None
        self.extra_action_pending = False

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
        print(
            f"{self.display_name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}"
        )

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
            f"{self.display_name} received {initial_hp - self.hp} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.display_name} has fainted!")

    def receive_damage(self, damage: int | float) -> None:
        if not self.alive:
            return

        # 检查是否有盾牌效果，如果有，应用伤害减免
        for status in self.statuses:
            if isinstance(status, effects.ShieldEffect):
                if status.duration == 1:
                    damage *= status.damage_multiplier
                    break  # 只应用一回合盾牌效果

        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.display_name}'s defense absorbed the attack!")
            return
        self.lose_hp(damage)

    def try_dodge(self, bonus: int = 0) -> bool:  # bonus参数用于增加闪避概率，默认为0
        if not self.alive:
            return False
        # 尝试闪避攻击
        if random.randint(1, 100) <= self.dodge_chance + bonus:
            print(f"{self.display_name} dodged the attack!")
            return True
        return False

    def on_damage_dealt(self) -> None:
        # 当造成伤害时触发的效果
        pass

    def can_act(self) -> bool:  # 处理麻痹效果，判断宝可梦是否可以行动
        if not self.alive:
            return False
        # 检查是否可以行动
        for status in self.statuses:
            if isinstance(status, effects.ParalysisEffect):
                if status.duration > 0:
                    status.apply(self)  # 应用麻痹效果
                    status.decrease_duration(self)  # 减少持续时间
                    if status.duration <= 0:
                        self.statuses.remove(status)
                    return False
        return True

    def add_status_effect(self, effect: Effect) -> None:
        effect_initial_duration = effect.duration
        # 检查是否已经有相同的状态效果，如果有，刷新持续时间
        if isinstance(effect, effects.ParalysisEffect):
            for status in self.statuses:
                if isinstance(status, effects.ParalysisEffect):
                    status.duration = max(status.duration, effect_initial_duration)
                    return
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        if not self.alive:
            return
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            if status.trigger_phase != "turn_end":
                continue  # 只在回合结束阶段应用状态效果
            status.apply(self)
            status.decrease_duration(self)
            if status.duration <= 0:
                print(f"{self.display_name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)
            if self.hp <= 0:  # hp小于0，结束循环
                break

    # 克制计算倍率由父类统一实现
    def type_effectiveness(self, opponent: Pokemon) -> float:
        # 根据对手的类型计算效果倍率
        return TYPE_EFFECTIVENESS.get(self.type, {}).get(opponent.type, 1.0)

    def begin(self) -> None:
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return (
            f"{self.display_name} type: {self.type} "
            f"HP: {self.hp}/{self.max_hp} speed: {self.speed}"
        )


# GrassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

    def begin(self) -> None:
        # 每个回合开始时执行草属性特性
        self.grass_attribute()

    def grass_attribute(self) -> None:
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        print(
            f"{self.display_name}'s Grass attribute activates! Healing 10% of max HP."
        )
        self.heal_self(amount)


# Bulbasaur 类，继承自 GrassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(
        self,
        hp: int = 100,
        attack: int = 35,
        defense: int = 10,
        dodge_chance: int = 10,
        *,
        speed: int = 45,
    ) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class WaterPokemon(Pokemon):
    type = "Water"

    # 水属性减伤 → 护盾减伤 → 取整 → 减防御 → 实际扣血
    def receive_damage(self, damage: int | float) -> None:
        if not self.alive:
            return
        # 受到伤害时，有 50% 的几率减免 30% 的伤害
        if random.randint(1, 100) <= 50:
            damage *= 0.7
            print(f"{self.display_name}'s Water attribute reduces the damage by 30%!")
        super().receive_damage(damage)


class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 25,
        defense: int = 20,
        dodge_chance: int = 20,
        *,
        speed: int = 43,
    ) -> None:
        # 初始化 Squirtle 的属性
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 AquaJet 和 Shield
        return [skills.AquaJet(), skills.Shield()]


class FirePokemon(Pokemon):
    type = "Fire"
    base_attack: int
    attack_stacks: int = 0  # 攻击力提升的层数

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_chance: int,
        *,
        speed: int = 50,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)
        self.base_attack = attack  # 保存基础攻击力
        self.attack_stacks = 0

    def on_switch_out(self) -> None:
        super().on_switch_out()
        self.attack_stacks = 0
        self.attack = self.base_attack

    def on_damage_dealt(self) -> None:
        if not self.alive:
            return
        if self.attack_stacks >= 4:
            return  # 攻击力提升最多为4层
        # 当造成伤害时触发的效果，增加攻击力
        self.attack_stacks += 1
        self.attack = int(
            self.base_attack * (1 + self.attack_stacks * 0.1)
        )  # 每次造成伤害增加10%的攻击力
        print(
            f"{self.display_name}'s attack increases to {self.attack} due to Fire attribute!"
        )


class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 15,
        dodge_chance: int = 10,
        *,
        speed: int = 65,
    ) -> None:
        # 初始化 Charmander 的属性
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 Ember 和 FlameCharge
        return [skills.Ember(), skills.FlameCharge()]


class ElectricPokemon(Pokemon):
    type = "Electric"

    def try_dodge(self, bonus: int = 0) -> bool:
        dodge_status = super().try_dodge(bonus)
        if dodge_status:
            # 如果闪避成功，触发电属性特性，增加一次额外行动机会
            self.extra_action_pending = True
        return dodge_status


class Pikachu(ElectricPokemon):
    name = "Pikachu"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 5,
        dodge_chance: int = 30,
        *,
        speed: int = 90,
    ) -> None:
        # 初始化 Pikachu 的属性
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 Thunderbolt 和 QuickAttack
        return [skills.Thunderbolt(), skills.QuickAttack()]


# 喵喵
# | 项目 | 设计 |
# |---|---|
# | 属性 | 普通 `Normal`，与现有属性双向倍率均为 `1.0` |
# | 基础数值 | HP **80**、攻击 **30**、防御 **10**、闪避 **20%** |
# | 被动：灵巧 | 当前 HP ≤ 最大 HP 的一半时，闪避额外增加 **10 个百分点**；回血超过一半后取消加成，不累计 |
# | 技能一：利爪 `Scratch` | 造成当前攻击力 **1.2 倍**的普通伤害，无附加效果 |
# | 技能二：聚宝功 `PayDay` | 造成当前攻击力 **0.8 倍**的普通伤害，恢复本次**实际扣血量的 50%**，向下取整 |
class NormalPokemon(Pokemon):
    type = "Normal"

    def try_dodge(self, bonus: int = 0) -> bool:
        # 检查当前 HP 是否低于最大 HP 的一半，若是则增加闪避率
        if self.hp <= self.max_hp / 2:
            bonus += 10
        return super().try_dodge(bonus)


class Meowth(NormalPokemon):
    name = "Meowth"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 30,
        defense: int = 10,
        dodge_chance: int = 20,
        *,
        speed: int = 90,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_chance, speed=speed)
        self.base_dodge_chance: int = dodge_chance  # 保存基础闪避率

    def initialize_skills(self) -> list[Skill]:
        # 初始化技能，具体技能是 Scratch 和 PayDay
        return [skills.Scratch(), skills.PayDay()]
