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
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}"
        )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} is poisoned by {self.name}!")
        else:
            print(f"{self.name} did not poison {opponent.name} this time.")


class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} heals {self.amount} HP with {self.name}")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} is poisoned by {self.name}!")


class Thunderbolt(Skill):
    name = "Thunderbolt"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = int(damage * 1.4)  # 1.4倍攻击力

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成电属性伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name} used {self.name}, dealing {self.damage} electric damage to {opponent.name}")

        # 10%概率使敌人麻痹
        if random.randint(1, 100) <= 10:
            opponent.add_status_effect(effects.ParalysisEffect(duration=1))
            print(f"{opponent.name} is paralyzed by {self.name}!")


class QuickAttack(Skill):
    name = "Quick Attack"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage  # 1.0倍攻击力

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成快速攻击伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}")

        # 10%概率触发第二次攻击
        if random.randint(1, 100) <= 10:
            opponent.receive_damage(self.damage)
            print(f"{self.name} hits again, dealing additional {self.damage} damage!")



class AquaJet(Skill):
    name = "Aqua Jet"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = int(damage * 1.4)  # 140%水属性伤害

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成水属性伤害
        opponent.receive_damage(self.damage)
        print(f"{user.name} used {self.name}, dealing {self.damage} water damage to {opponent.name}")


class Shield(Skill):
    name = "Shield"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 减少下一回合受到的伤害50%
        user.add_status_effect(effects.ProtectEffect(duration=1))
        print(f"{user.name} used {self.name}, reducing next turn's damage by 50%!")

