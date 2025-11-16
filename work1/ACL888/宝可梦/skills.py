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
class ThunderShock(Skill):
    name = "Thunder Shock"
    #对敌人造成 1.0 倍攻击力的快速攻击（快速攻击有几率触发第二次攻击），10% 概率触发第二次
    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}"
        )
        # 判断是否触发第二次攻击
        if random.randint(1, 100) <= 10:
            opponent.receive_damage(self.damage)
            print(
                f"{user.name} used {self.name} again, dealing {self.damage} damage to {opponent.name}"
            )
class millionvolits(Skill):
    name = "millionvolits"
    #对敌人造成 1.4 倍攻击力的电击攻击，10% 概率使对手麻痹

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}"
        )

        # 判断是否触发麻痹效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalyzeEffect())
            print(f"{opponent.name} is paralyzed by {self.name}!")
        else:
            print(f"{self.name} did not paralyze {opponent.name} this time.")
class watergun(Skill):
    name = "Water Gun"
    #对敌人造成 1.0 倍攻击力的水枪攻击

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}"
        )
class Shield(Skill):
    name = "Shield"
    #给予自身护盾效果

    def __init__(self) -> None:
        super().__init__()  
    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        #引入effects中的defenseeffect
        user.add_status_effect(effects.DefenseEffect())
        print(f"{user.name} used {self.name} and gained a shield.")
class Ember(Skill):
    name = "Ember"
    #对敌人造成 1 倍攻击力的火焰攻击，10% 概率使对手灼伤

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} used {self.name}, dealing {self.damage} damage to {opponent.name}"
        )

        # 判断是否触发灼伤效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.burnEffect())
            print(f"{opponent.name} is burned by {self.name}!")
        else:
            print(f"{self.name} did not burn {opponent.name} this time.")
class FlameCharge(Skill):
    name = "Flame Charge"
    #

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 两回合蓄力：第一回合蓄力并记录将要释放的动作，第二回合自动释放
        # 不在本回合造成伤害或灼伤，而是在下一回合释放
            # 第一次使用：开始蓄力，不立刻造成强力伤害；蓄力期结束时由 ChargeEffect 执行释放伤害
            # 防止在同一回合或已有 ChargeEffect 时重复蓄力
            if any(getattr(s, "name", None) == "Charge" for s in getattr(user, "statuses", [])):
                print(f"{user.name} is already charging and cannot use {self.name} again.")
                return

            # 创建 ChargeEffect，保存目标和最终伤害（比如 3x damage）
            # duration=2 表示：当前回合结束时减为1，下一回合开始时 apply 生效，下一回合结束时释放
            charged_damage = int(self.damage * 3)
            user.add_status_effect(effects.ChargeEffect(amount=1, damage=charged_damage, target=opponent, duration=2))
            print(f"{user.name} is charging up for {self.name}! Will deal {charged_damage} damage when charged.")
            # 触发灼伤判定立即添加灼伤（原需求）
            if random.randint(1, 100) <= 80:
                opponent.add_status_effect(effects.burnEffect())
                print(f"{opponent.name} is burned by {self.name}!")
            else:
                print(f"{self.name} did not burn {opponent.name} this time.")
        