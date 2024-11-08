from __future__ import annotations
import skills
from skills import Skill
from effects import Effect


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses = []

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name} 使用了 {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 恢复了 {amount} 点生命值! 当前生命值: {self.hp}/{self.max_hp}")



    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}的防御抵御了伤害")
            return

        self.hp -= damage
        print(
            f"{self.name} 受到了 {damage} 点基础伤害! 当前生命值: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 进入休眠状态!")



    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}的 {status.name} 状态消失了")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} 类型: {self.type}"


# GlassPokemon 类
class GlassPokemon(Pokemon):
    type = "Glass"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率

        opponent_type = opponent.type

        if opponent_type == "Water":
            self.damage*=2
        elif opponent_type == "Fire":
            self.damage%=0.5
        return self.damage

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
            f"{self.name} 恢复了 {amount} 点生命! 当前生命: {self.hp}/{self.max_hp}"
        )
# FirePokemon 类
class FirePokemon(Pokemon):
    type = ("Fire")

    def type_effectiveness(self, opponent: Pokemon):

        # 针对敌方 Pokemon 的类型，调整效果倍率


        opponent_type = opponent.type

        if opponent_type == "Glass":
            self.damage*=2
        elif opponent_type == "Water":
            self.damage*=0.5
        return self.damage




    def fire_attribute(self):
        self.attack_boost_layers = 0
        self.attack_boost_layers_max = 4


        self.attack_boost_layers+=1
        if self.attack_boost_layers < self.attack_boost_layers_max:
            self.attack = self.attack * 1.1



# Bulbasaur(妙蛙种子) 类，继承自 GlassPokemon
class Bulbasaur(GlassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=35, defense=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=35), skills.ParasiticSeeds(amount=10)]

# Charmander(小火龙) 类，继承自 FirePokemon
class Charmander(FirePokemon):
    name = "Charmander"
    def __init__(self, hp=80, attack=35, defense=15) -> None:
        # 初始化 Charmander 的属性
        super().__init__(hp, attack, defense)

    def initialize_skills(self):
        # 初始化技能，具体技能是 Ember 和 Flame Charge
        return [skills.Ember(damage=35), skills.Flame_Charge(duration=2)]