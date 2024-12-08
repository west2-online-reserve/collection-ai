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



class Thunderbolt(Skill):
    name = "Thunderbolt"
    def __init__(self, damage_multiplier: float = 1.4) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * self.damage_multiplier
        opponent.receive_damage(int(damage))
        # 10% 概率使敌人麻痹
        if random.random() < 0.1:
            print(f"{opponent.name} is paralyzed!")
            opponent.add_status_effect(effects.ParalyzeEffect(duration=2)) 

class QuickAttack(Skill):
    name = "Quick Attack"
    def __init__(self, damage_multiplier: float = 1.0) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * self.damage_multiplier
        opponent.receive_damage(int(damage))
        # 10% 概率触发第二次攻击
        if random.random() < 0.1:
            print(f"{user.name} triggers a second Quick Attack!")
            opponent.receive_damage(int(damage))  # 进行第二次攻击

class AquaJet(Skill):
    name = "AquaJet"
    def __init__(self, damage_multiplier: float = 1.4) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * self.damage_multiplier
        opponent.receive_damage(int(damage))

class Shield(Skill):
    name = "SHield"
    def __init__(self) -> None:
        super().__init__()
        self.shield_active = False

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}! Reducing damage by 50% next turn.")
        user.add_status_effect(effects.ShieldEffect(duration=1))  # 调用


class Ember(Skill):
    name = "Ember"
    def __init__(self, damage_multiplier: float = 1.0) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * self.damage_multiplier
        opponent.receive_damage(int(damage))
        # 10% 概率使目标陷入“烧伤”状态
        if random.random() < 0.1:
            print(f"{opponent.name} is burned!")
            opponent.add_status_effect(effects.BurnEffect(duration=2, burn_damage=10))  # 假设有烧伤效果


class FlameCharge(Skill):
    name = "FlameCharge"
    def __init__(self, damage_multiplier: float = 3.0) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier
        self.requires_charge = True  # 蓄力
        self.evasion_increase = 0.2  

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if self.requires_charge:
            print(f"{user.name} is charging up for {self.name}! Opponent's evasion rate increases by 20%.")
            opponent.evasion_rate += self.evasion_increase  # 增加对手的闪避率
            self.requires_charge = False  # 蓄力完成
        else:
            print(f"{user.name} unleashes {self.name}!")
            damage = user.attack * self.damage_multiplier
            opponent.receive_damage(int(damage))
            self.requires_charge = True  # 下次使用需要重新蓄力
            opponent.evasion_rate -= self.evasion_increase  # 恢复对手的闪避率
            # 80% 概率使目标陷入“烧伤”状态
            if random.random() < 0.8:
                print(f"{opponent.name} is burned!")
                opponent.add_status_effect(effects.BurnEffect(duration=2, burn_damage=10))


class Gust(Skill):
    name = "Gust"
    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * 1.0
        opponent.receive_damage(int(damage))

class AirSlash(Skill):
    name = "AirSlash"
    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"{user.name} uses {self.name}!")
        damage = user.attack * 1.5
        opponent.receive_damage(int(damage))