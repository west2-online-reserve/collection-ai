import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


def printf(web):
    print(web)
    # time.sleep(0.3)
    input()


class Skill:
    name: str

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class attSkill(Skill):
    name: str

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        if random.randint(1, 100) > opponent.avoid:
            printf(f"{opponent.name}'s avoid is failed!")
            self.do_execute(user, opponent)
        else:
            printf(f"{opponent.name} aviod the attack")
            if opponent.type == "Electric":
                opponent.avoid = True
            return

    def do_execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        raise NotImplementedError


class SeedBomb(attSkill):
    name = "Seed Bomb"

    def __init__(self, activation_chance: int = 15) -> None:
        super().__init__()
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def do_execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        amount = user.attack
        # 造成伤害
        eff = user.type_effectiveness(opponent)  # 计算特攻
        opponent.receive_damage(amount, True, eff)
        # printf(
        #     f"{user.name} used {self.name}, dealing {hplose} damage to {opponent.name}"
        # )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            printf(f"{opponent.name} is poisoned by {self.name}!")
        else:
            printf(f"{self.name} did not poison {opponent.name} this time.")


class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect())
        printf(f"{user.name} heals 10 HP with {self.name}")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        printf(f"{opponent.name} is poisoned by {self.name}!")


class Ember(attSkill):
    name = "Ember"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def do_execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(amount, True, eff)
        # printf(
        #             f"{user.name} used {self.name}, dealing {hplose} damage to {opponent.name}"
        #         )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.fire())
            printf(f"{opponent.name} is fired!")
        else:
            printf(f"{self.name} is not fired {opponent.name} this time.")


class FlameChargePrepare(Skill):
    name = "Flame Charge prepare"

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        printf(f"{user.name} is preparing Flame Charage")
        # if not any(isinstance(skill,FlameChargePrepare) for skill in user.skills) :
        #     raise ValueError("技能不存在")
        user.skills = [s for s in user.skills if not isinstance(s, FlameChargePrepare)]
        user.skills.append(FlameChargeReady())


class FlameChargeReady(attSkill):
    name = "Flame Charge"

    def __init__(self, activation_chance: int = 80):
        super().__init__()
        self.activation_chance = activation_chance

    def do_execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(amount * 3.0, True, eff)
        # printf(
        #             f"{user.name} used {self.name}, dealing {hplose} damage to {opponent.name}"
        #         )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.fire())
            printf(f"{opponent.name} is fired!")
        else:
            printf(f"{self.name} is not fired {opponent.name} this time.")

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not any(isinstance(skill, FlameChargeReady) for skill in user.skills):
            raise ValueError("技能不存在")
        user.skills = [s for s in user.skills if not isinstance(s, FlameChargeReady)]
        user.skills.append(FlameChargePrepare())
        if random.randint(1, 100) > opponent.avoid + 20:
            printf(f"{opponent.name}'s avoid is failed!")
            self.do_execute(user, opponent)
        else:
            printf(f"{opponent.name} aviod the attack")
            return


class AquaJet(attSkill):
    name = "Aqua Jet"

    def __init__(self):
        super().__init__()

    def do_execute(self, user, opponent):
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(amount, True, eff)


class Shield(Skill):
    name = "Shield"

    def __init__(self):
        super().__init__()

    def execute(self, user, opponent):
        user.add_status_effect(effects.Shield())


class Thunderbolt(attSkill):
    name = "Thunderbolt"

    def __init__(self):
        self.activation_chance = 10
        super().__init__()

    def do_execute(self, user, opponent):
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(amount * 1.4, True, eff)
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.paralysis())
            printf(f"{opponent.name} is paralyzed!")
        else:
            printf(f"{self.name} is not paralyzed {opponent.name} this time.")


class QuickAttack(attSkill):
    name = "Quick Attack"

    def __init__(self):
        self.activation_chance = 10
        super().__init__()

    def do_execute(self, user, opponent):
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(amount * 1.4, True, eff)
        if random.randint(1, 100) <= self.activation_chance:
            print(f"{user.name} attack angin")
            opponent.receive_damage(amount * 1.4, True, eff)
        else:
            printf(f"{self.name} don't attack {opponent.name} this time.")


class RockFist(attSkill):
    name = "Rock Fist"

    def __init__(self):
        super().__init__()

    def do_execute(self, user, opponent):
        amount = user.attack
        eff = user.type_effectiveness(opponent)
        opponent.receive_damage(
            amount * (1 + (user.max_hp - user.hp) * 2 / user.max_hp), True, eff
        )


class RageBuild(Skill):
    name = "Rage Build"

    def __init__(self):
        super().__init__()

    def execute(self, user, opponent):
        user.hp *= 0.7
        print(f"{user.name}'hp lost to {user.hp:.1f}/{user.max_hp},it'Rage is bulit")
        user.add_status_effect(effects.Rageattup())
        user.add_status_effect(effects.Ragedefup())
        user.add_status_effect(effects.RageDRup())
