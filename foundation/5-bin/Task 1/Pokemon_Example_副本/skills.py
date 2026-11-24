import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str
    priority: int = 0

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    name = "Seed Bomb"

    def __init__(self, activation_chance: int = 15) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        # 计算伤害
        skill_damage = user.attack * user.type_effectiveness(opponent)
        opponent_initial_hp = opponent.hp
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
        if not opponent.alive:
            return  # 如果对手已经被击败，则不再触发状态效果
        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.display_name} is poisoned by {self.name}!")
        else:
            print(
                f"{user.display_name}'s {self.name} did not poison "
                f"{opponent.display_name} this time."
            )


class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        # 给对手添加寄生种子效果
        opponent.add_status_effect(effects.LeechSeedEffect(source=user))


class AquaJet(Skill):
    name = "Aqua Jet"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害
        skill_damage = user.attack * 1.4 * user.type_effectiveness(opponent)
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法


class Shield(Skill):
    name = "Shield"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给自己添加护盾效果
        user.add_status_effect(effects.ShieldEffect())
        print(f"{user.display_name} used {self.name}")


class Ember(Skill):
    name = "Ember"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害
        skill_damage = user.attack * 1.0 * user.type_effectiveness(opponent)
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
        if not opponent.alive:
            return  # 如果对手已经被击败，则不再触发状态效果
        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())


class FlameCharge(Skill):
    name = "Flame Charge"

    def __init__(self, activation_chance: int = 80) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive or not user.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if user.charging_skill is not self:
            print(f"{user.display_name} is charging {self.name} and will attack next turn!")
            user.charging_skill = self
            return
        else:
            print(f"{user.display_name} unleashes the charged {self.name}!")
            user.charging_skill = None  # 重置蓄力状态
        if opponent.try_dodge(bonus=20):  # 增加20%的闪避概率
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害
        skill_damage = user.attack * 3.0 * user.type_effectiveness(opponent)
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
        if not opponent.alive:
            return  # 如果对手已经被击败，则不再触发状态效果
        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())


class Thunderbolt(Skill):
    name = "Thunderbolt"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害
        skill_damage = user.attack * 1.4 * user.type_effectiveness(opponent)
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
        if not opponent.alive:
            return  # 如果对手已经被击败，则不再触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())


class QuickAttack(Skill):
    name = "Quick Attack"
    priority = 1

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive or not user.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害，一般属性技能，不考虑克制关系
        skill_damage = user.attack * 1.0
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
        if not opponent.alive or not user.alive:
            return  # 如果对手已经被击败，则不再触发追击效果
        # 判断是否触发追击效果
        if random.randint(1, 100) <= self.activation_chance:
            print(f"{user.display_name} triggers a follow-up attack with {self.name}!")
            if opponent.try_dodge():
                return
            opponent_initial_hp = opponent.hp
            skill_damage = user.attack * 1.0  # 追击伤害与普通攻击相同
            opponent.receive_damage(skill_damage)
            if opponent_initial_hp - opponent.hp > 0:
                user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法


class Scratch(Skill):
    name = "Scratch"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive or not user.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害，一般属性技能，不考虑克制关系
        skill_damage = user.attack * 1.2
        opponent.receive_damage(skill_damage)
        if opponent_initial_hp - opponent.hp > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法


class PayDay(Skill):
    name = "PayDay"

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if not opponent.alive or not user.alive:
            return
        print(f"{user.display_name} used {self.name}")
        if opponent.try_dodge():
            return
        opponent_initial_hp = opponent.hp
        # 计算伤害，一般属性技能，不考虑克制关系
        skill_damage = user.attack * 0.8
        opponent.receive_damage(skill_damage)
        actual_damage_dealt = opponent_initial_hp - opponent.hp
        if actual_damage_dealt > 0:
            user.on_damage_dealt()  # 触发使用者的 on_damage_dealt 方法
            heal_amount = actual_damage_dealt // 2  # 向下取整
            user.heal_self(heal_amount)
