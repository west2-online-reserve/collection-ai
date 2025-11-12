import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str

    def __init__(self) -> None:
        pass
    # user和opponent都是Pokemon类的实例
    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"

# PikaChu的技能
class Thunderbolt(Skill):
    name = "Thunderbolt"

    def __init__(self, damage_multiplier: float = 1.4, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier
        self.activation_chance = activation_chance


    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.attack = int(user.attack * self.damage_multiplier)
        actual_damage = user.calculate_damage(user.type, opponent.type, user, opponent)
        print(f"{user.name}使用了{self.name}，造成了{actual_damage}点伤害")
        opponent.receive_damage(actual_damage)

        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalyzeEffect())
        else:
            print(f"{opponent.name}没有被麻痹")

class QuickAttack(Skill):
    name = "Quick Attack"

    def __init__(self, activation_chance: int = 10):
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage = user.calculate_damage(user.type, opponent.type, user, opponent)
        print(f"{user.name}使用了{self.name}，造成了{actual_damage}点伤害")
        opponent.receive_damage(actual_damage)

        if random.randint(1, 100) <= self.activation_chance:
            print("触发了第二次攻击")
            actual_damage = user.calculate_damage(user.type, opponent.type, user, opponent)
            print(f"{user.name}使用了{self.name}，造成了{actual_damage}点伤害")
            opponent.receive_damage(actual_damage)

class SeedBomb(Skill):
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        actual_damage = user.calculate_damage(user.type, opponent.type, user, opponent)
        print(f"{user.name}使用了{self.name}，造成了{actual_damage}点伤害")
        opponent.receive_damage(actual_damage)

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            print(f"{opponent.name}陷入中毒状态")
            opponent.add_status_effect(effects.PoisonEffect())
        else:
            print(f"{opponent.name}没有陷入中毒状态")


class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        amount = 0.1 * opponent.max_hp
        opponent.hp -= amount
        user.hp += amount
        if user.hp >= user.max_hp:
            user.hp = user.max_hp
        print(f"{opponent.name}减少了{amount}点血量，{user.name}恢复了{amount}点血量")


# Squirtle的技能
class AquaJet(Skill):
    name = "Aqua Jet"

    def __init__(self, damage_multiplier: float = 1.4) -> None:
        super().__init__()
        self.damage_multiplier = damage_multiplier

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        original_attack = user.attack
        user.attack = int(user.attack * self.damage_multiplier)
        actual_damage = user.calculate_damage(user.type, opponent.type, user, opponent)
        print(f"{user.name}使用了{self.name}，造成了{actual_damage}点伤害")
        opponent.receive_damage(actual_damage)
        user.attack = original_attack

class Shield(Skill):
    name = "Shield"

    def __init__(self):
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.add_status_effect(effects.ShieldEffect())

# Charmander的技能
class Ember(Skill):
    name = "Ember"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        attack_damage = user.attack
        print(f"{user.name}使用了{self.name}")

        opponent.receive_damage(attack_damage)
        print(f"{opponent.name}受到了{attack_damage}点伤害")

        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}陷入了烧伤状态")

class FlameCharge(Skill):
    name = "Flame Charge"

    def __init__(self, damage_mutiplier: int = 3, activation_chance: int = 80, is_charging: bool = True) -> None:
        super().__init__()
        self.damage_mutiplier = damage_mutiplier
        self.activation_chance = activation_chance
        self.is_charging = is_charging

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if self.is_charging == False:
            attack_damage = int(user.attack * self.damage_mutiplier)
            print(f"{user.name}使用了{self.name}")

            opponent.receive_damage(attack_damage)
            print(f"{opponent.name}受到了{attack_damage}点伤害")

            if random.randint(1, 100) <= self.activation_chance:
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.name}陷入了烧伤状态")

            self.is_charging = True
        else:
            user.add_status_effect(effects.ChargeEffect())
            print(f"{user.name}开始蓄力...")
            self.is_charging = False


