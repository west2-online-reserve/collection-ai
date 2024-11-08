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
        # 为子类留下接口,提醒后续还有操作
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"

#  "种子炸弹"
class SeedBomb(Skill):
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: int = 100) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害

        opponent.receive_damage(self.damage)


        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            # 用随机数来实现触发几率
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} 触发了 {self.name} 的技能状态!")



        else:
            print(f"{self.name} 本轮没有触发 {opponent.name} 的技能状态")

# "寄生种子"
class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} 恢复了 {self.amount} 点血量 ")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} is poisoned by {self.name}!")

# "火花"
class Ember(Skill):
    name = "Ember"

    def __init__(self, damage: int, burn_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.burn_chance = burn_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.burn_chance:
            # 用随机数来实现触发几率
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} 触发了 {self.name} 的技能状态 !")
        else:
            print(f"{self.name} 本轮没有触发 {opponent.name} 的技能状态")


# "蓄能爆炎"
class Flame_Charge(Skill):
    name = "Flame Charge"

    def __init__(self, damage: int=35, burn_chance: int = 80,duration: int = 3) -> None:
        super().__init__()
        self.damage = damage
        self.burn_chance = burn_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        self.damage = self.damage * 2  # 蓄能状态下伤害翻倍
        opponent.receive_damage( self.damage)
        self.damage = self.damage / 2  # 回合结束后伤害恢复正常


        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.burn_chance:
            # 用随机数来实现触发几率
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} 触发了 {self.name} 的技能状态 !")
        else:
            print(f"{self.name} 本轮没有触发 {opponent.name} 的技能状态")



