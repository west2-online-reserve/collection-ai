import effects
import random


class skill:
    name: str

    def __init__(self) -> None:
        pass

    def execute(self, user, opponent):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(skill):
    name = "Seed Bomb"

    def __init__(self, damage: int, activation_chance: float = 0.15) -> None:
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user, opponent) -> None:
        # 造成伤害
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
        )

        # 判断是否触发状态效果
        if random.random() < self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.name} 被 {self.name} 毒化了！")
        else:
            print(f"{self.name} 没有对 {opponent.name} 造成中毒效果。")


class ParasiticSeeds(skill):
    name = "Parasitic Seeds"

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def execute(self, user, opponent) -> None:
        # 给使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"{user.name} 使用了 {self.name} 治疗了 {self.amount} HP")

        # 给对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"{opponent.name} 被 {self.name} 毒化了！")


class Thunderbolt(skill):
    name = "Thunderbolt"

    def __init__(self, damage: int, chance: float = 0.1) -> None:
        super().__init__()
        self.damage = damage
        self.chance = chance

    def execute(self, user, opponent) -> None:
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
        )
        if random.random() < self.chance:
            opponent.add_status_effect(effects.Paralysis())
            print(f"{opponent.name} 被麻痹了！")


class Quickattack(skill):
    name = "Quickattack"

    def __init__(self, damage: int, chance: float = 0.1) -> None:
        super().__init__()
        self.damage = damage
        self.chance = chance

    def execute(self, user, opponent):
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
        )
        if random.random() < self.chance:
            self.execute(user, opponent)  # 触发后再次攻击一次
            print(f"{self.name} 触发了二段攻击！")


class AquaJet(skill):
    name = "AquaJet"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user, opponent):
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
        )


class Shield(skill):
    name = "Shield"

    def __init__(self, reduce: float) -> None:
        super().__init__()
        self.reduce = reduce

    def execute(self, user, opponent):
        user.reduce = self.reduce
        print(f"{user.name} 使用了 {self.name}, 伤害将减少 {int(self.reduce*100)}%")


class Ember(skill):
    name = "Ember"

    def __init__(self, damage: int, chance: float = 0.1) -> None:
        super().__init__()
        self.damage = damage
        self.chance = chance

    def execute(self, user, opponent):
        opponent.receive_damage(self.damage)
        print(
            f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
        )
        if random.random() < self.chance:
            opponent.add_status_effect(effects.Burn())
            print(f"{opponent.name} 被烧伤了！")


class FlameCharge(skill):
    name = "Flame Charge"

    def __init__(self, damage: int, reduce: float) -> None:
        super().__init__()
        self.damage = damage
        self.reduce = reduce

    def execute(self, user, opponent):
        if user.count == 0:
            user.count += 1
            return
        else:
            opponent.dodge_rate += self.reduce
            opponent.receive_damage(self.damage)
            if opponent.dodge_is_successful:
                print("被闪开了！")
                user.count = 0
                return
            else:
                print(
                    f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
                )
                user.count = 0


class Echoism(skill):
    name = "Echoism"

    # 叠层，不造成任何伤害，如果位于麻痹状态就额外叠层
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def execute(self, user, opponent):
        if opponent.nerve_damage:
            print(f"{opponent.name} 已经处于神经损伤状态，无法再叠加了！")
            return
        else:
            stack_gain = self.amount
            opponent.nerve_count += stack_gain
            print(
                f"{user.name} 使用了 {self.name}, 使 {opponent.name} 的神经损伤层数增加了 {stack_gain} 层！当前层数: {opponent.nerve_count}"
            )
            opponent.nerve_is_successful()


class Mass(skill):
    name = "Mass"

    # 造成一次伤害，如果对方位于神经损伤状态额外造成伤害
    def __init__(self, damage: int, extra_damage: int):
        super().__init__()
        self.damage = damage
        self.extra_damage = extra_damage

    def execute(self, user, opponent):
        if opponent.nerve_damage:
            total_damage = self.damage + self.extra_damage
            opponent.receive_damage(total_damage)
            print(
                f"{user.name} 使用了 {self.name}, 造成了 {total_damage} 点伤害给 {opponent.name}（包括神经损伤的额外伤害）"
            )
        else:
            opponent.receive_damage(self.damage)
            print(
                f"{user.name} 使用了 {self.name}, 造成了 {self.damage} 点伤害给 {opponent.name}"
            )
