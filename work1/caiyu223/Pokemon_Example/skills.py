import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str

    def __init__(self) -> None:
        self.skill_type = ''
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    name = "种子炸弹"
    #技能在宝可梦技能表中的位次
    skill_number = 1
    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 对 {opponent.name} 造成了 {self.damage} 点伤害"
        )

        # 判断是否触发状态效果
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} 被 {self.name} 中毒了!")
        else:
            print(f"{self.name} 这次没有让 {opponent.name} 中毒。")


class ParasiticSeeds(Skill):
    name = "寄生种子"
    skill_number = 2

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} 使用 {self.name} 恢复了 {self.amount} 点生命值")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} 被 {self.name} 中毒了!")


class Thunderbolt(Skill):
    name = '十万伏特'
    skill_number = 1

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        opponent.receive_damage(1.4*user.attack)
        if random.randint(1,100) <= 10:
            print('麻痹成功，对方的攻击似乎更容易躲避了')
            opponent.add_status_effect(effects.ParalysisEffect())

        
class QuickAttack(Skill):
    name = '电光一闪'
    skill_number = 2

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        opponent.receive_damage(user.attack)
        if random.randint(1,100) <= 10:
            print('第二次攻击')
            opponent.receive_damage(user.attack)


class AquaJet(Skill):
    name = '水流喷射'
    skill_number = 1

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        opponent.receive_damage(1.4*user.attack)

class Shield(Skill):
    name = '护盾'
    skill_number = 2

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.shield = True
        print(f"{user.operator}的 {user.name} 使用了护盾")

class Ember(Skill):
    name = '火花'
    skill_number = 1

    def execute(self, user: 'Pokemon', opponent: 'Pokemon') -> None:
        opponent.receive_damage(user.attack)
        if random.randint(1,100) <= 10:
            opponent.add_status_effect(effects.BurnEffect())

class FlameCharge(Skill):
    name = "蓄能焰袭"
    skill_number = 2

    def __init__(self) -> None:
        self.skill_type = 'delay'
    
    def execute(self, user: 'Pokemon', opponent: 'Pokemon') -> None:
        #释放阶段
        if user.delay_skill == self:
            print('爆炎')
            opponent.receive_damage(user.attack*3)
            if random.randint(1,100) <= 80:
                opponent.add_status_effect(effects.BurnEffect())
            return
        #延迟阶段
        user.delay_skill = self
        print('开始蓄能')