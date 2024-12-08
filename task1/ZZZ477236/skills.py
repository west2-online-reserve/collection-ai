import random
from typing import TYPE_CHECKING
import effects
from westt.effects import Effect

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
    name = "种子炸弹"

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 计算伤害并考虑属性影响因子
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")

        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name}被{self.name}中毒了！")


class ParasiticSeeds(Skill):
    name = "寄生种子"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name}使用了{self.name}，恢复了{self.amount}点HP")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name}被{self.name}中毒了！")


class Thunderbolt(Skill):
    name = "十万伏特"

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")

        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name}被{self.name}麻痹了！")


class QuickAttack(Skill):
    name = "电光一闪"

    def __init__(self, damage: int, extra_attack_chance: int = 10) -> None:
        super().__init__()
        self.damage = damage
        self.extra_attack_chance = extra_attack_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")

        if random.randint(1, 100) <= self.extra_attack_chance:
            print(f"{user.name}触发了额外攻击！")
            opponent.receive_damage(total_damage)
            print(f"{opponent.name}受到了来自{user.name}的快速攻击的额外{total_damage}点伤害！")


class Ember(Skill):
    name = "火花"

    def __init__(self, damage: int, burn_chance: int = 10, burn_damage: int = 10, burn_duration: int = 2) -> None:
        super().__init__()
        self.damage = damage
        self.burn_chance = burn_chance
        self.burn_damage = burn_damage
        self.burn_duration = burn_duration

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")

        if random.randint(1, 100) <= self.burn_chance:
            opponent.add_status_effect(effects.BurnEffect(self.burn_damage, self.burn_duration))
            print(f"{opponent.name}被{self.name}灼伤了！")


class FlameCharge(Skill):
    name = "蓄能爆炎"

    def __init__(self, damage: int, burn_chance: int = 80, burn_duration: int = 2, charge_turns: int = 1,
                 evasion_increase: int = 20) -> None:
        super().__init__()
        self.damage = damage
        self.burn_chance = burn_chance
        self.burn_duration = burn_duration
        self.charge_turns = charge_turns
        self.evasion_increase = evasion_increase

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if self.charge_turns > 0:
            opponent.evasion_rate += self.evasion_increase
            self.charge_turns -= 1
            print(f"{user.name}正在为{self.name}蓄力！")
            print(f"{opponent.name}的闪避率增加到{opponent.evasion_rate}%！")
        else:
            total_damage = self.damage * user.type_effectiveness(opponent)
            opponent.receive_damage(total_damage)
            print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")

            if random.randint(1, 100) <= self.burn_chance:
                opponent.add_status_effect(effects.BurnEffect(10, self.burn_duration))
                print(f"{opponent.name}被{self.name}灼伤了！")
            else:
                print(f"{self.name}未能使{opponent.name}灼伤。")


class AquaJet(Skill):
    name = "水枪"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{user.name}使用了{self.name}，对{opponent.name}造成了{total_damage}点伤害")


class Shield(Skill):
    name = "水盾"

    def __init__(self, duration: int = 1) -> None:
        super().__init__()
        self.duration = duration

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.add_status_effect(effects.ShieldEffect(self.duration))
        print(f"{user.name}使用了{self.name}，激活了持续{self.duration}回合的护盾。")


class WaterGun(Skill):
    name = "水弹"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{opponent.name}受到了来自{self.name}的{total_damage}点水属性伤害！")


class Confusion(Skill):
    name = "纷乱头脑"

    def __init__(self, damage: int, confusion_chance: int = 20) -> None:
        super().__init__()
        self.damage = damage
        self.confusion_chance = confusion_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        total_damage = self.damage * user.type_effectiveness(opponent)
        opponent.receive_damage(total_damage)
        print(f"{opponent.name}因{self.name}受到了{total_damage}点伤害")

        if random.randint(1, 100) <= self.confusion_chance:
            opponent.add_status_effect(effects.ConfusionEffect())
            print(f"{opponent.name}陷入了混乱！")
