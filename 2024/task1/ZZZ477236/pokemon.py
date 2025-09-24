from __future__ import annotations
import skills
from effects import Effect
from skills import Skill
import random
class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int,evasion_rate: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.evasion_rate = evasion_rate
        self.base_attack=attack
        self.skills = self.initialize_skills()
        self.alive = True
        self.shield_active=False
        self.statuses = []

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        #print(f"{self.name} 使用了 {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 恢复了 {amount} 点体力! 剩余体力: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}防御住了这次攻击！")
            return

        self.hp -= damage
        print(
            f"{self.name} 受到了 {damage} 点伤害! 剩余体力: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 已经失去了所有体力！")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name} 的 {status.name} 状态效果已经失效！")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def remove_status_effect(self, effect: Effect) -> None:
        # 从状态效果列表中移除指定效果
        if effect in self.statuses:
            self.statuses.remove(effect)
            print(f"{effect.name} 状态效果已从 {self.name}身上移除！")

    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"


# GlassPokemon 类
class GlassPokemon(Pokemon):
    type = "草"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "水":
            effectiveness = 2.0
        elif opponent_type == "火":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        # 每个回合开始时执行草属性特性
        self.glass_attribute()

    def glass_attribute(self):
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} 在回合开始时恢复了 {amount} 点体力! 剩余体力: {self.hp}/{self.max_hp}"
        )


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GlassPokemon):
    name = "妙蛙种子"

    def __init__(self, hp=100, attack=50, defense=10,evasion_rate=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense,evasion_rate)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=40), skills.ParasiticSeeds(amount=10)]



class ElectricityPokemon(Pokemon):
    type = "电"

    def type_effectiveness(self, opponent: "Pokemon"):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "水":
            effectiveness = 2.0
        elif opponent_type == "草":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        self.electricity_attribute()

    def electricity_attribute(self):
        if random.randint(1, 100) < self.evasion_rate:
            print(f"{self.name} 成功躲避了这次攻击并将随机释放一次技能!")
            skills = self.skills
            if skills:  # 确保技能列表不为空
                random_skill = random.choice(skills)
                print(f"{self.name} 随机使用了 {random_skill}!")
                self.use_skill(random_skill, self)
            else:
                print("无计可施")


class Pikachu(ElectricityPokemon):
    name = "皮卡丘"

    def __init__(self, hp=80, attack=35, defense=21, evasion_rate=30) -> None:
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self):
        return [skills.Thunderbolt(damage=int(self.attack * 1.4)), skills.QuickAttack(damage=self.attack)]


# FirePokemon 类
class FirePokemon(Pokemon):
    type = "火"

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "草":
            effectiveness = 2.0
        elif opponent_type == "水":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        self.fire_attribute()

    def fire_attribute(self):
        # 火属性特性：每次造成伤害后增加攻击力，最多叠加40%基础攻击力
        attack_boost = self.base_attack * 0.1
        self.attack += min(attack_boost, self.base_attack * 0.4)
        print(f"{self.name}的攻击力增加了 {attack_boost}!")

class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self, hp=80, attack=35, defense=20, evasion_rate=10) -> None:
        # 初始化
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self):
        # 初始化技能 火花和蓄能爆炎
        return [skills.Ember(damage=35, burn_chance=20), skills.FlameCharge(damage=55, burn_chance=80)]


class WaterPokemon(Pokemon):
    type = "水"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "火":
            effectiveness = 2.0
        elif opponent_type == "电":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        self.water_attribute()

    def water_attribute(self):
        # 水属性特性：受到伤害时，有50%的几率减免30%的伤害
        if random.randint(1, 100) <= 50:
            print(f"{self.name} 触发了水属性特性：减少30％的伤害")
            return 0.7
        return 1.0

class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, hp=80, attack=25, defense=25, evasion_rate=20) -> None:
        super().__init__(hp, attack, defense, evasion_rate)

    def initialize_skills(self):
        return [skills.AquaJet(damage=38), skills.Shield()]

class Psyduck(WaterPokemon):
    name = "可达鸭"
    def __init__(self,hp=70,attack=28, defense=22, evasion_rate=20)-> None:
        super().__init__(hp,attack,defense ,evasion_rate)
    def initialize_skills(self):
        return[skills.WaterGun(35),skills.Confusion(30)]