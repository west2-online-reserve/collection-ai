from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.sheild=0
        self.skills = self.initialize_skills()
        self.effective=1.0
        self.alive = True
        self.statuses = []

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name}使用了{skill.name}")
        skill.execute(self, opponent)
    
    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name}回复了{amount}血量， 当前血量： {self.hp}/{self.max_hp}")
    def add_shield(self, amount):
        # 为自身添加护盾
        if not isinstance(amount, int):
            amount = int(amount)
        self.sheild += amount
        print(f"{self.name}获得了{amount}的护盾")
    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)
        damage-=self.sheild
        damage -= self.defense
        damage*=self.effective
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return

        self.hp -= damage
        print(
            f"{self.name}受到了{damage}点伤害，当前血量：{self.hp if self.hp>=0 else 0.0}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name}已被击败")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}的{status.name}效果消失了")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"


# GlassPokemon 类
class GrassPokemon(Pokemon):
    type = "草"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "水":
            effectiveness = 0.5
        elif opponent_type == "火":
            effectiveness = 2.0
        return effectiveness

    def begin(self):
        # 每个回合开始时执行玻璃属性特性
        self.glass_attribute()

    def glass_attribute(self):
        # 玻璃属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name}回复了{amount}血量， 当前血量： {self.hp if self.hp>=0 else 0.0}/{self.max_hp}"
        )


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"

    def __init__(self, hp=100, attack=35, defense=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]
class WaterPokemon(Pokemon):
    type = "水"
    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type
        if opponent_type == "草":
            effectiveness = 2.0
        elif opponent_type == "火":
            effectiveness = 0.5
        return effectiveness
    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return
        self.hp -= damage
        #水属性特性
        if random.randint(1, 100) <= 50:
            damage=damage*0.7
        print(
            f"{self.name}受到了{damage}点伤害，当前血量：{self.hp if self.hp>=0 else 0.0}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name}已被击败")
class Squirtle(WaterPokemon):
    name = "杰尼龟"
    def __init__(self, hp=80, attack=25, defense=20) -> None:
        # 初始化属性
        super().__init__(hp, attack, defense)
    def initialize_skills(self):
        return [skills.AquaJet(damage=45), skills.Shield(amount=10)]

class FirePokemon(Pokemon):
    type = "火"
    def __init__(self, hp=80, attack=35, defense=15,passive=0) -> None:
        # 初始化属性
        super().__init__(hp, attack, defense)
        self.passive=passive
    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type
        if opponent_type == "草":
            effectiveness = 0.5
        elif opponent_type == "水":
            effectiveness = 2.0
        return effectiveness
    def use_skill(self, skill, opponent):
        #火系被动
        if self.passive<4:
            self.attack+=5
            self.passive+=1
            print(f"{self.name}的攻击力增加了5点")
        return super().use_skill(skill, opponent)
class Charmander(FirePokemon):
    name = "小火龙"
    def __init__(self, hp=80, attack=35, defense=15,passive=1) -> None:
        # 初始化属性
        super().__init__(hp, attack, defense,passive)
    def initialize_skills(self):
        return [skills.Ember(damage=self.attack), skills.FlameThrower(damage=30)]