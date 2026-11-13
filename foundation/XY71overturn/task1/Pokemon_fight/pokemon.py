from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random
import math

class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int,dodge_rate:float,charging_skill:bool=None,defense_rate:float=0.0,skip_turn:bool=False) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.dodge_rate=dodge_rate
        self.charging_skill = charging_skill
        self.defense_rate=defense_rate
        self.skip_turn=skip_turn
        self.skills = self.initialize_skills()
        self.alive = True
        self.base_attack = attack
        self.flame_stack = 0  # 用于追踪火焰属性的叠加攻击力
        self.statuses = []

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def dodge_action(self,opponent: Pokemon) -> bool:
        # 计算是否闪避攻击
        dodge_rate = self.dodge_rate
        if opponent.name == "Charmander" and opponent.charging_skill is True:
            dodge_rate += 0.2
        return random.random() < min(dodge_rate,1.0)

    def damage_multiplier(self)->float:
        return 1.0

    def use_skill(self, skill: Skill, opponent: Pokemon,allow_counter:bool=True):
        # 使用技能
        if not self.alive or not opponent.alive:
            return
        if self.skip_turn:
            print(f"{self.name} is paralyzed and cannot move this turn!")
            return
        print(f"{self.name} uses {skill.name}")
        if  skill.can_be_dodged(self) and opponent.dodge_action(self):
            print(f"{opponent.name} dodged the attack!")
            if opponent.type=="Flash" and allow_counter:
                print(f"{opponent.name} counterattacks!")
                opponent.use_skill(random.choice(opponent.skills),self,allow_counter=False)
            if skill.name=="Flame Charge":
                self.charging_skill=None
            return
        
        else:
            hp_before = opponent.hp
            skill.execute(self, opponent)

            if self.type == "Fire" and opponent.hp < hp_before:
                self.flame_attribute()
        

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        hp_before = self.hp
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        actual_heal = self.hp - hp_before
        print(f"{self.name} heals {actual_heal} HP! Current HP: {self.hp}/{self.max_hp}")
        return actual_heal

    def receive_damage(self, damage, attacker: Pokemon | None = None):
        # 直接攻击先计算攻击者对自身的属性克制；持续伤害不传攻击者。
        if attacker is not None:
            damage *= attacker.type_effectiveness(self)
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)
        if attacker is not None:
            damage -= self.defense
            damage = damage*(1-self.defense_rate)
            damage *=self.damage_multiplier()
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return 0

        hp_before = self.hp
        self.hp -= damage
        self.hp = max(self.hp, 0)  # 确保 HP 不会低于 0
        actual_damage = hp_before - self.hp  # 实际造成的伤害值
        print(
            f"{self.name} received {actual_damage} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
        return actual_damage  # 返回实际造成的伤害值

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        if effect.name == "Paralysis":
            for status in self.statuses:
                if status.name == "Paralysis":
                    print(f"{self.name} is already paralyzed!")
                    status.duration = max(status.duration, effect.duration)  # 更新持续时间为较长的一个
                    return
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        self.skip_turn=False  # 重置跳过回合标志
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            if not self.alive:
                break
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                if status.name == "Paralysis" and self.skip_turn:
                    print(f"{self.name}'s {status.name} duration has ended, but it can not act this turn due to paralysis.")
                else:
                    print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


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
        # 每个回合开始时执行玻璃属性特性
        self.glass_attribute()

    def glass_attribute(self):
        # 玻璃属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        hp_before = self.hp
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        actual_heal = self.hp - hp_before
        print(
            f"{self.name} heals {actual_heal} HP at the start of the turn! Current HP: {self.hp}/{self.max_hp}"
        )

# FlamePokemon 类
class FlamePokemon(Pokemon):
    type = "Fire"

    def type_effectiveness(self, opponent: Pokemon):

        effectiveness=1.0
        opponent_type = opponent.type

        if opponent_type == "Glass":
            effectiveness = 2.0
        elif opponent_type == "Water":
            effectiveness = 0.5
        return effectiveness


    def flame_attribute(self):
        # 火焰属性特性：每次造成伤害，叠加 10% 攻击力，最多 4 层
        if self.flame_stack>=4:
            return
        else:
            self.flame_stack += 1
            self.attack = self.base_attack * (1 + 0.1 * self.flame_stack)
            print(f"{self.name}'s flame attribute increased attack to {self.attack}!")

#WaterPokemon 类
class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Fire":
            effectiveness = 2.0
        elif opponent_type == "Flash":
            effectiveness = 0.5
        return effectiveness

    def damage_multiplier(self)->float:
        if random.random() < 0.5:
            print(f"{self.name}'s water attribute reduced damage!")
            return 0.7
        return 1.0

# FlashPokemon 类
class FlashPokemon(Pokemon):
    type = "Flash"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Glass":
            effectiveness = 2.0
        elif opponent_type == "Water":
            effectiveness = 0.5
        return effectiveness


    
# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GlassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=35, defense=10,dodge_rate=0.1) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=35), skills.ParasiticSeeds(amount=10)]


# Charmander 类，继承自 FlamePokemon
class Charmander(FlamePokemon):
    name = "Charmander"

    def __init__(self,hp = 80,attack = 35,defense = 15,dodge_rate=0.1,charging_skill=None) -> None:
        # 初始化 Charmander 的属性
        super().__init__(hp, attack, defense, dodge_rate, charging_skill)
        self.charging_skill = charging_skill


    def initialize_skills(self):
        return[skills.Ember(),skills.Flame_Charge()]

# Squirtle 类，继承自 WaterPokemon
class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20,dodge_rate=0.2) -> None:
        # 初始化 Squirtle 的属性
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        return [skills.WaterGun(damage=35), skills.AquaRing(defense_rate=0.5)]

# Pikachu 类，继承自 FlashPokemon
class Pikachu(FlashPokemon):
    name = "Pikachu"

    def __init__(self, hp=80, attack=35, defense=5,dodge_rate=0.3) -> None:
        # 初始化 Pikachu 的属性
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        return [skills.Thunderbolt(damage=49), skills.Quick_Attack(damage=35)]


# Hertzwing 类，继承自 FlashPokemon
class Hertzwing(FlashPokemon):
    name = "Hertzwing"

    def __init__(self, hp=75, attack=40, defense=5,dodge_rate=0.35) -> None:
        # 初始化 Hertzwing 的属性
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self):
        return [skills.Re_Volt(damage=32), skills.Shockupuncture(damage=60)]
