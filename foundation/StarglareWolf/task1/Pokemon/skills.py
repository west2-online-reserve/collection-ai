import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon
# 因为 pokemon 文件中已经导入了 skills 和 effects，这里如果直接导入 pokemon, 会出现 A 导入 B, 然后 B 导入 A 的情况。在 C++ 中可以，因为是分开编译，但是 python 中 import 相当于直接编译那个文件，相互嵌套导入就会死局

class Skill:
    name: str

    def __init__(self, damage) -> None:
        self.damage = damage
        self.can_release = True
        self.need_cool = False
        self.cool_round = 0
        pass

    # 执行
    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def decrease_cool_round(self):
        pass


    def change_damage(self, user: "Pokemon", opponent: "Pokemon") -> int:
        damage = self.damage * (1 + user.super_position_times * 0.1)
        damage *= user.type_effectiveness(opponent)
        if opponent.passive_defense_success:
            damage *= 0.7
        if opponent.protective_shield:
            damage *= 0.5
            opponent.protective_shield = False    # 对方的防护盾减免起效后就关闭，这样不必改变出战顺序也能实现比较公平
        if not isinstance(damage, int):
            damage = int(damage)
        return damage

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    # Pokemon 用引号引起来是为了延迟求值，防止出现作用域不存在该类名。因为导入 Pokemon 使用的是 TYPE_CHECKING
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        if opponent.dodge_success():
            print(f"{opponent.name} 躲开了这次攻击!")
            return
        damage = self.change_damage(user, opponent)
        opponent.receive_damage(damage)

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} 陷入中毒状态!")
        else:
            pass


class ParasiticSeeds(Skill):
    name = "Parasitic Seeds"

    def __init__(self, amount: int) -> None:
        super().__init__(0)
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} 恢复 {self.amount} HP!")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} 陷入中毒状态!")

class Thunderbolt(Skill):
    name = "Thunderbolt"

    def __init__(self, damage, activation_chance: int = 10) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance


    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if opponent.dodge_success():
            print(f"{opponent.name} 躲开了这次攻击!")
            return
        damage = self.change_damage(user, opponent)
        damage *= 1.4
        if not isinstance(damage, int) :
            damage = int(damage)

        opponent.receive_damage(damage)

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name} 陷入麻痹状态!")
        else:
            pass


class QuickAttack(Skill):
    name = "Quick Attack"
    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if opponent.dodge_success():
            print(f"{opponent.name} 躲开了这次攻击!")
            return

        damage = self.change_damage(user, opponent)
        opponent.receive_damage(damage)

        # 判断是否触发状态效果，再次攻击对方
        if random.randint(1, 100) <= self.activation_chance:
            print(f"{user.name} 成功发动二次进攻!", end = "\t")
            opponent.receive_damage(damage)
        else:
            pass


class Aquajet(Skill):
    name = "Aqua jet"
    def __init__(self, damage: int, activation_chance: int = 100) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if opponent.dodge_success():
            print(f"{opponent.name} 躲开了这次攻击!")
            return
        damage = self.change_damage(user, opponent)
        damage *= 1.4    # 造成 140%伤害
        if not isinstance(damage, int) :
            damage = int(damage)
        opponent.receive_damage(damage)

class Shield(Skill):
    name = "Shield"
    def __init__(self, damage: int, activation_chance: int = 100) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if random.randint(1, 100) <= self.activation_chance:
            user.protective_shield = True
            print(f"{user.name} 开启了保护罩,抵御对方 50% 进攻!")
        else:
            user.protective_shield = False

class Ember(Skill):
    name = "Ember"
    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if opponent.dodge_success():
            print(f"{opponent.name} 躲开了这次攻击!")
            return
        damage = self.change_damage(user, opponent)   # 100% 攻击伤害
        opponent.receive_damage(damage)

        # 烧伤状态判定
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnedEffect())


class Flamecharge(Skill):
    name = "Flame Charge"
    def __init__(self, damage: int, need_cool = True, activation_chance: int = 80) -> None:
        super().__init__(damage)
        self.activation_chance = activation_chance
        self.cool_round = 2
        self.need_cool = need_cool

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if self.can_release:
           self.cool_round = 2
        if opponent.dodge_success(20):  # 敌人闪避概率增加 20%
            print(f"{opponent.name} 躲开了这次攻击!")
            return
        damage = self.change_damage(user, opponent)
        damage *= 3  # 300% 火属性伤害
        opponent.receive_damage(damage)

        # 烧伤状态判定条件
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnedEffect())

    def decrease_cool_round(self):
        self.cool_round -= 1
        if self.cool_round <= 0:
            self.can_release = True
        else:
            self.can_release = False




class Blank(Skill):
    name = "空白攻击"
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        print(f"发动了{self.name}攻击!")
