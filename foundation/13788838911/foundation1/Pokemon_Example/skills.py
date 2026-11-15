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
        multiplier = user.type_effectiveness(opponent)
        final_damage = int(self.damage * multiplier)
        
        opponent.receive_damage(final_damage)
        print(
            f"{user.name} used {self.name}, dealing {final_damage} damage to {opponent.name}"
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

    def __init__(self,damage: int=49,activation_chance:int =10)->None:
        super().__init__()
        self.damage=damage
        self.activation_chance=activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
            # 造成伤害
            finaldamage=user.type_effectiveness(opponent)*self.damage
            opponent.receive_damage(finaldamage)
            print(
                f"{user.name} used {self.name}, dealing {finaldamage} damage to {opponent.name}"
            )
    
            # 判断是否触发状态效果
            if random.randint(1, 100) <= self.activation_chance:
                opponent.add_status_effect(effects.PoisonEffect())
                print(f"{opponent.name} is paralyzed by {self.name}!")
            else:
                print(f"{self.name} did not paralyzed {opponent.name} this time.")

class QuickAttack(Skill):
    name = "Quick_attack"

    def __init__(self,damage: int,activation_chance:int =10)->None:
        super().__init__()
        self.damage=damage
        self.activation_chance=activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
            # 造成伤害
            finaldamage=user.type_effectiveness(opponent)*self.damage
            opponent.receive_damage(finaldamage)
            print(
                f"{user.name} used {self.name}, dealing {finaldamage} damage to {opponent.name}"
            )
    
            # 判断是否触发状态效果
            if random.randint(1, 100) <= self.activation_chance:
                if not opponent.alive:
                     return 
                print(f"{self.name} hits again!")
                opponent.receive_damage(finaldamage)
                print(f"{opponent.name} takes additional {finaldamage} damage!")
            else:
                print(f"{self.name} did not trigger a second hit.")

class AquaJet(Skill):
    name = "Aqua Jet"

    def __init__(self,damage: int=49)->None:
        super().__init__()
        self.damage=damage
        

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
            # 造成伤害
            finaldamage=user.type_effectiveness(opponent)*self.damage
            opponent.receive_damage(finaldamage)
            print(
                f"{user.name} used {self.name}, dealing {finaldamage} damage to {opponent.name}"
            )

class Shield(Skill):
    name = "Shield"

    def __init__(self,amount:float=0.5)->None:
        super().__init__()
        self.amount=amount
        
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
            # 造成伤害
        user.add_status_effect(effects.ShieldEffect(self.amount))
        print(f"{user.name} raised a shield! Damage reduced by {int(self.amount*100)}% next turn.")


class Ember(Skill):
    name="Ember"

    def __init__(self,damage: int=35,activation_chance:int =10)->None:
            super().__init__()
            self.damage=damage
            self.activation_chance=activation_chance
    
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
            # 造成伤害
            finaldamage=user.type_effectiveness(opponent)*self.damage
            opponent.receive_damage(finaldamage)
            print(
            f"{user.name} used {self.name}, dealing {finaldamage} damage to {opponent.name}"
                )
        
                # 判断是否触发状态效果
            if random.randint(1, 100) <= self.activation_chance:
                   opponent.add_status_effect(effects.BurnEffect())
                   print(f"{opponent.name} is burned by {self.name}!")
            else:
                print(f"{self.name} did not burned {opponent.name} this time.")

class FlameCharge(Skill):
    name = "Flame Charge"

    def __init__(self, damage: int = 35, activation_chance: int = 80) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance
        self.charging = False

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if self.charging:
             finaldamage = int(self.damage *3* user.type_effectiveness(opponent))
             opponent.receive_damage(finaldamage)
             print(f"{user.name} used {self.name}, dealing {finaldamage} damage to {opponent.name}")

             if random.randint(1, 100) <= self.activation_chance:
                  opponent.add_status_effect(effects.BurnEffect(duration=2))
                  print(f"{opponent.name} is burned by {self.name}!")
             else:
                  print(f"{self.name} did not burn {opponent.name} this time.")
             self.charging = False
        else:
             self.charging =True
             print(f"{user.name} is charging {self.name}!")