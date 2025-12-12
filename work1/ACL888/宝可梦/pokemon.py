from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int,dodge_possibility:float ) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.dodge_possibility = dodge_possibility
        # 初始化技能由子类实现
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses = []
        # 状态标志
        self.paralyzed = False
        self.charging = False
        self.charge = 0
    # 两回合蓄力相关：仅使用状态效果 ChargeEffect 来管理蓄力与释放，移除重复字段

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能（技能自身负责打印使用信息以避免重复）
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    def dodge(self, possibility: float) -> bool:
        """Try to dodge an incoming attack. Returns True if dodged."""
        if random.random() < possibility:
            print(f"{self.name} dodged the attack!")
            return True
        return False

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)
        # 先尝试闪避（基于固有闪避概率）
        if hasattr(self, "dodge_possibility") and self.dodge_possibility:
            if self.dodge(self.dodge_possibility):
                return

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return

        self.hp -= damage
        print(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        # 仅应用效果（不在此处减少持续时间）——持续时间在 end_turn 中处理
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)

    def end_turn(self):
        """Tick down all status durations and handle expirations. Call at end of a turn."""
        for status in self.statuses[:]:
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                try:
                    status.on_expire(self)
                except Exception:
                    pass
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        # 在回合开始时先应用所有状态效果（持续时间在回合结束时扣减）
        self.apply_status_effect()

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


# GlassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

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
        # 每个回合开始时执行草属性特性
        # 先应用状态效果（如中毒/灼伤），再执行草属性回血
        super().begin()
        self.grass_attribute()

    def grass_attribute(self):
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} heals {amount} HP at the start of the turn! Current HP: {self.hp}/{self.max_hp}"
        )


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=50, defense=10, dodge_possibility=0.2) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense, dodge_possibility)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=35), skills.ParasiticSeeds(amount=10)]
class lightningpokemon(Pokemon):
    type = "lightning"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "water":
            effectiveness = 2.0
        elif opponent_type == "grass":
            effectiveness = 0.5
        return effectiveness
    
    def dodge_attack(self):
    #闪避一次攻击立刻发动一次攻击
     if "dodge" in [status.name for status in self.statuses]:
         print(f"{self.name} dodged the attack and counterattacks!")
         self.statuses = [status for status in self.statuses if status.name != "dodge"]
         return True
    
class Pikachu(lightningpokemon):
    name = "Pikachu"

    def __init__(self, hp=90, attack=60, defense=15) -> None:
        # 初始化 Pikachu 的属性
        super().__init__(hp, attack, defense,dodge_possibility=0.3)

    def initialize_skills(self):
        # 初始化技能，具体技能是 ThunderShock 和 millionvolits
        return [skills.millionvolits(damage=49), skills.ThunderShock(damage=35)]
class waterpokemon(Pokemon):
    type = "water"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "fire":
            effectiveness = 2.0
        elif opponent_type == "lightning":
            effectiveness = 0.5
        return effectiveness
    
class Squirtle(waterpokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20, dodge_possibility=0.2) -> None:
        # 初始化 Squirtle 的属性
        super().__init__(hp, attack, defense, dodge_possibility)

    def initialize_skills(self):
            # 初始化技能，具体技能是 WaterGun 和 Shield
            return [skills.watergun(damage=35), skills.Shield( )]
class firepokemon(Pokemon):
    type = "fire"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "grass":
            effectiveness = 2.0
        elif opponent_type == "water":
            effectiveness = 0.5
        return effectiveness
    
class Charmander(firepokemon):
    name = "Charmander"

    def __init__(self, hp=80, attack=35, defense=15, dodge_possibility=0.1) -> None:
        # 初始化 Charmander 的属性
        super().__init__(hp, attack, defense, dodge_possibility)

    def initialize_skills(self):
            # 初始化技能，具体技能是 Ember 和 FlameCharge
            return [skills.Ember(damage=35), skills.FlameCharge(damage=35)]
    

