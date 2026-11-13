import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str
    dodgeable = False

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"

    def can_be_dodged(self,user:"Pokemon")-> bool:
        return self.dodgeable


class SeedBomb(Skill):
    name = "Seed Bomb"
    dodgeable=True

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        print(
            f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}"
        )

        if not opponent.alive:
            return  # 如果对手已经被击败，则不再尝试添加状态效果
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
        heal_amount = opponent.max_hp * self.amount // 100
        user.add_status_effect(effects.HealEffect(heal_amount))
        print(f"{user.name} gains a healing effect from {self.name}.")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} is poisoned by {self.name}!")

class Ember(Skill):
    name = "Ember"
    dodgeable=True

    def __init__(self,activation_chance:int=10) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self,user:"Pokemon",opponent:"Pokemon") -> None:
        actual_damage=opponent.receive_damage(user.attack, attacker=user)
        print(f"{user.name} used {self.name},dealing {actual_damage} to {opponent.name}")

        if not opponent.alive:
            return  # 如果对手已经被击败，则不再尝试添加状态效果
        if random.randint(1,100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name} is burned by {self.name}!")
        else:
            print(f"{self.name} did not burn {opponent.name} this time.")

class Flame_Charge(Skill):
    name ="Flame Charge"

    def can_be_dodged(self, user:"Pokemon")->bool:
        return user.charging_skill is True

    def __init__(self,activation_chance:int=80) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self,user:"Pokemon",opponent:"Pokemon") -> None:
        if user.charging_skill is None:
            user.charging_skill = True
            print(f"{user.name} is charging up for a powerful attack!")
        elif user.charging_skill==True:
            actual_damage=opponent.receive_damage(user.attack*3, attacker=user)
            print(f"{user.name} used {self.name},dealing {actual_damage} to {opponent.name}")
            
            user.charging_skill = None
            if not opponent.alive:
                return  # 如果对手已经被击败，则不再尝试添加状态效果
            if random.randint(1,100) <= self.activation_chance:
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.name} is burned by {self.name}!")
            else:
                print(f"{self.name} did not burn {opponent.name} this time.")
            

class WaterGun(Skill):
    name = "Water Gun"
    dodgeable=True

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}")

class AquaRing(Skill):
    name = "Aqua Ring"

    def __init__(self, defense_rate: float,duration: int=1) -> None:
        super().__init__()
        self.defense_rate = defense_rate
        self.duration = duration

    def execute(self, user:"Pokemon", opponent:"Pokemon")->None:
        user.defense_rate=self.defense_rate
        print(f"{user.name} used {self.name}, received a shield.")

class Thunderbolt(Skill):
    name = "Thunderbolt"
    dodgeable=True

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}")

        if not opponent.alive:
            return  # 如果对手已经被击败，则不再尝试添加状态效果

        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name} is paralyzed by {self.name}!")
        else:
            print(f"{self.name} did not paralyze {opponent.name} this time.")        

class Quick_Attack(Skill):
    name = "Quick Attack"
    dodgeable=True

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}")

        if random.randint(1, 100) <= self.activation_chance:
            if opponent.alive:
                actual_damage=opponent.receive_damage(self.damage, attacker=user)
                print(f"{self.name} activated again, dealing an additional {actual_damage} damage to {opponent.name}!")
        else:
            print(f"{self.name} did not activate again this time.")

class Re_Volt(Skill):
    name = "Re-Volt"
    dodgeable=True

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        if not opponent.alive:
            print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}. {opponent.name} has fainted!")
            return
        actual_damage+=opponent.receive_damage(self.damage, attacker=user)
        print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}")

class Shockupuncture(Skill):
    name = "Shockupuncture"
    dodgeable=True

    def __init__(self, damage: int, activation_chance: int = 30) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        actual_damage=opponent.receive_damage(self.damage, attacker=user)
        print(f"{user.name} used {self.name}, dealing {actual_damage} damage to {opponent.name}")

        if not opponent.alive:
            return  # 如果对手已经被击败，则不再尝试添加状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name} is paralyzed by {self.name}!")
        else:
            print(f"{self.name} did not paralyze {opponent.name} this time.")
        
