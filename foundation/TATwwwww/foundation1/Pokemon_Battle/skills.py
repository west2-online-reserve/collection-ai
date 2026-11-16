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
        dmg = user.calculate_outgoing_damage(self.damage, opponent, "Grass")
        opponent.receive_damage(dmg)
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

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        dmg = user.calculate_outgoing_damage(user.attack * 1.4, opponent, "Electric")
        opponent.receive_damage(dmg)
        print(
            f"{user.name} used {self.name}, dealing {user.attack*1.4} damage to {opponent.name}"
        )
        if opponent.alive and random.randint(1, 100) <= 10:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name} is paralyzed by {self.name}!")
        else:
            print(f"{self.name} did not paralyze {opponent.name} this time.")

class QuickAttack(Skill):
    name = "Quick Attack"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        dmg = user.calculate_outgoing_damage(user.attack, opponent, "Electric")
        opponent.receive_damage(dmg)
        print(
            f"{user.name} used {self.name}, dealing {user.attack} damage to {opponent.name}"
        )
        if opponent.alive and random.randint(1, 100) <= 10:
            opponent.receive_damage(user.attack)
            print(f"{opponent.name} is hit again by {self.name}, dealing {user.attack} damage!")

class AquaJet(Skill):
    name = "Aquajet"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        dmg = user.calculate_outgoing_damage(user.attack * 1.4, opponent, "Water")
        opponent.receive_damage(dmg)
        print(
            f"{user.name} used {self.name}, dealing {user.attack * 1.4} damage to {opponent.name}"
        )

class Shield(Skill):
    name = "Shield"

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        user.shield = True
        print(f"{user.name} raises its shield with {self.name}")

class Ember(Skill):
    name = "Ember"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        dmg = user.calculate_outgoing_damage(user.attack, opponent, "Fire")
        opponent.receive_damage(dmg)
        print(
            f"{user.name} used {self.name}, dealing {user.attack} damage to {opponent.name}"
        )
        if opponent.alive and random.randint(1, 100) <= 10:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} is burned by {self.name}!")
        else:
            print(f"{self.name} did not burn {opponent.name} this time.")

class FlameCharge(Skill):
    name = "Flame Charge"
    #小火龙召唤出强大的火焰，对敌人造成 300% 火属性伤害，并有 80% 的几率使敌人陷入「烧伤」状态。
    #这个技能需要 1 个回合蓄力，并且在面对该技能时敌方闪避率增加 20%
    def execute(self, user, opponent):
        if user.charging_skill is None:
            user.charging_skill = self
            print(f"{user.name} is charging! It will release next turn!")
            return

        user.charging_skill = None

        if random.random() < opponent.dodge_rate + 0.2:
            print(f"{opponent.name} dodged Flame Charge!")
            opponent.on_dodge(user)
            return

        dmg = user.calculate_outgoing_damage(user.attack * 3.0, opponent, "Fire")
        opponent.receive_damage(dmg)
        user.on_deal_damage()
        if opponent.alive and random.randint(1, 100) <= 80:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} is burned!")

class Dig(Skill):
    name = "Dig"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        dmg = user.calculate_outgoing_damage(user.attack * 1.5, opponent, "Ground")
        opponent.receive_damage(dmg)
        print(f"{user.name} used {self.name}, dealing {dmg} damage to {opponent.name}")

class SandAttack(Skill):
    name = "Sand Attack"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        opponent.attack = int(opponent.attack * 0.9)
        print(f"{opponent.name}'s attack falls by {self.name}!")