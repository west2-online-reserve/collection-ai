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
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    name = "种子炸弹"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name}使用了{self.name}, 对{opponent.name}造成了{self.damage}点伤害"
        )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name}中毒了!")
        else:
            print(f"{self.name}本次没有对{opponent.name}造成中毒效果")


class ParasiticSeeds(Skill):
    name = "寄生种子"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name}回复了{self.amount}点血量")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} 中毒了！")
class AquaJet(Skill):
    name = "水枪"
    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name}使用了{self.name}, 对{opponent.name}造成了{self.damage}点伤害")
class Shield(Skill):
    name = "护盾"
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加护盾效果
        user.add_status_effect(effects.ShieldEffect(self.amount))
#小火龙技能
class Ember(Skill):
    name = "火花"
    def __init__(self, damage: int,activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name}使用了{self.name}, 对{opponent.name}造成了{self.damage}点伤害")
        #判断是否烧伤
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}烧伤了!")
class FlameThrower(Skill):
    name = "蓄能爆炎"
    def __init__(self, damage: int,activation_chance: int = 80) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name}使用了{self.name}, 对{opponent.name}造成了{self.damage}点伤害")
        #判断是否烧伤
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}烧伤了!")
