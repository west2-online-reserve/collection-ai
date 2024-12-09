import random
from typing import TYPE_CHECKING

import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError # 用于指示某个功能或方法尚未实现。它通常用在抽象基类或接口中，表明子类应该实现这个方法

    def __str__(self) -> str:
        return f"{self.name}"


# HP: 100 攻击力: 35 防御力: 10 属性: 草 躲闪率: 10%
#
# **种子炸弹 (Seed Bomb)：**妙蛙种子发射一颗种子，爆炸后对敌方造成草属性伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失10%生命值
#
# **寄生种子 (Parasitic Seeds)：**妙蛙种子向对手播种，每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合
class SeedBomb(Skill):
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化
   # skill.execute(self, opponent)
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:# 前向引用:以避免在定义之前引用该类型
        # 造成伤害
        #print(f"{user.name}对{opponent.name}发动技能" )

        print(
            f"{user.name} used {self.name}, dealing {user.type_effectiveness(opponent) * self.damage} damage to {opponent.name}"
        )
        opponent.receive_damage(user.type_effectiveness(opponent) * self.damage)
        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect(damage = 10))
            print(f"{opponent.name} is poisoned by {self.name}!")
        else:
            print(f"{self.name} did not poison {opponent.name} this time.")
        # print("\n")


# class ParasiticSeeds(Skill):
#     name = "Parasitic Seeds"
#
#     def __init__(self, amount: int) -> None:
#         super().__init__()
#         self.amount = amount
#
#     def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
#         # 给使用者添加治疗效果
#         print("-" * 20 + "技能发动" + "-" * 20)
#         self.amount *= user.type_effectiveness(opponent)
#         user.add_status_effect(effects.HealEffect(self.amount))
#         print(f"{user.name} heals {self.amount} HP with {self.name}")
#
#         # 给对手添加中毒效果
#         opponent.add_status_effect(effects.PoisonEffect())
#         print(f"{opponent.name} is poisoned by {self.name}!")
#
#         print("\n")


#寄生种子 (Parasitic Seeds)
class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self) -> None:
        super().__init__()
        # self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        #print(f"{user.name}发动技能造成影响")
        # self.amount *= user.type_effectiveness(opponent)
        amount = opponent.max_hp * 0.1 * user.type_effectiveness(opponent)
        user.add_status_effect(effects.HealEffect(amount = amount))
        opponent.add_status_effect(effects.VampireEffect(amount = amount))
        print(f"{user.name} heals {amount} HP with {self.name}")
        print(f"{opponent.name} loses {amount} HP by {self.name}")
        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect(damage = amount))
        print(f"{opponent.name} is poisoned by {self.name}!")

        #print("\n")

# **水枪 (Aqua Jet)：**杰尼龟喷射出一股强力的水流，对敌方造成 140% 水属性伤害
#
# **护盾 (Shield)：**杰尼龟使用水流形成保护盾，减少下一回合受到的伤害50%
class AquaJet(Skill):
    name = "Aqua Jet"
    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:


        self.damage *= user.type_effectiveness(opponent)
        print(f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}")
        opponent.receive_damage(self.damage)
        # print("\n")


class Shield(Skill):
    name = "Shield"
    def __init__(self, ability: float = 0.5) -> None:
        super().__init__()
        self.ability = ability
    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        #print("-" * 20+"技能发动"+"-" * 20)
        # user.protect_active= True
        print(f"{user.name} is on guard")
        print(f"opponent {opponent.name}  receive no damage! Remaining HP: {opponent.hp}/{opponent.max_hp}")
        user.add_status_effect(effects.ProtectEffect(user.type_effectiveness(opponent) * self.ability))
        # print("\n")



# **十万伏特 (Thunderbolt)：**对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人麻痹
#
# **电光一闪 (Quick Attack)：**对敌人造成 1.0 倍攻击力的快速攻击（快速攻击有几率触发第二次攻击），10% 概率触发第二次
class Thunderbolt(Skill):
    name = "Thunder bolt"
    def __init__(self, damage: int,activation_chance :int ) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:

        # 给对手添加伤害
        print(f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}")#1.4 * 35 =49
        opponent.receive_damage(self.damage)


        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            # opponent.add_status_effect(effects.ParalysisEffect())
            user.paralysis = True
            print(f"{opponent.name} is paralysised by {self.name}!")
        else:
            print(f"{self.name} did not paralysis {opponent.name} this time.")
        # print("\n")




class QuickAttack(Skill):
    name = "QuickAttack"
    def __init__(self,damage: int,activation_chance: int) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance
    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        opponent.receive_damage(self.damage)
        if random.randint(1, 100) <= self.activation_chance:
            opponent.receive_damage(self.damage)
