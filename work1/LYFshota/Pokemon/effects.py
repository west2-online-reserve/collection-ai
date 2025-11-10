from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon

class Effect:
    name: str  # 效果名称
    level: int = 1  # 层数
    duration: int  # 持续时间（回合数）
    max_level: int | None = None  # 最大层数（可选）

    def __init__(self, duration: int, level: int | None = None) -> None:

        self.duration = duration  # 初始化效果持续时间

        # 当 level 为 None 时视为未显式传入（仅刷新持续时间的场景）；
        # 当 level 非 None 时视为显式传入，传入 1 也被视为显式。
        self._explicit_level = level is not None

        # 初始化层数：如果传入的层数无效或小于1，则强制为1；如果未显式传入，则默认层数为1
        if level is None:
            lvl = 1
        else:
            try:
                lvl = int(level)
            except Exception:
                lvl = 1
        self.level = lvl if lvl >= 1 else 1

    def apply(self, pokemon: "Pokemon") -> None:
        # pokemon: 效果作用的目标宝可梦
        raise NotImplementedError

    def decrease_duration(self, pokemon: "Pokemon") -> None:
        # 每回合结束时调用，使效果持续时间减少1回合
        self.duration -= 1
        print(f"{pokemon.team}的{pokemon.name} {self.name}效果剩余回合数: {self.duration}")

    def refresh(self, duration: int, level: int = None) -> None:
        # 刷新效果的持续时间
        # 如果未传入层数，则视为传入层数为1；如果传入了层数，则将传入层数叠加到当前层数
        # 无论是否传入层数，都会刷新持续时间为传入的 duration
        if level is None:
            increment = 1
        else:
            try:
                increment = int(level)
            except Exception:
                increment = 1
            if increment < 1:
                increment = 1

        # 将传入的层数叠加到当前层数
        self.level += increment
        # 更新持续时间
        self.duration = duration


# 中毒效果类
class PoisonEffect(Effect):
    name = "中毒" 

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        # 初始化中毒效果
        # damage: 每回合造成的伤害百分比
        # duration: 效果持续回合数，默认为3回合
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用中毒效果，每回合对宝可梦造成固定伤害
        # pokemon: 被施加中毒效果的宝可梦
        # 中毒按当前生命的百分比造成伤害（忽略防御）
        # 传入的 self.damage 被视为百分比整数
        pre_hp = pokemon.hp

        try:
            ratio = float(self.damage) / 100.0
        except Exception:
            ratio = 0.1

        # 至少造成 1 点伤害
        damage = max(1, int(pre_hp * ratio))

        pokemon.hp -= damage
        if pokemon.hp <= 0:
            pokemon.hp = max(0, pokemon.hp)
            try:
                pokemon.alive = False
            except Exception:
                pass

        print(f"{pokemon.team}的{pokemon.name} 受到 {pre_hp - pokemon.hp} 点中毒伤害（{int(ratio*100)}% 当前生命）！剩余生命: {max(0, pokemon.hp)}/{pokemon.max_hp}")


# 治疗效果类
class HealEffect(Effect):
    name = "治疗" 

    def __init__(self, amount: int, duration: int = 3) -> None:
        # 初始化治疗效果
        # amount: 每回合恢复的生命值
        # duration: 效果持续回合数，默认为3回合
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用治疗效果，每回合为宝可梦恢复固定生命值
        # pokemon: 被施加治疗效果的宝可梦
        pokemon.heal_self(self.amount)
        print(f"{pokemon.team}的{pokemon.name} 触发治疗效果")

# 麻痹效果类
class ParalyzeEffect(Effect):
    name = "麻痹" 

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用麻痹效果：将目标宝可梦在本回合设为无法行动
        try:
            pokemon.can_act = False
        except Exception:
            # 如果目标对象没有 can_act 属性，忽略
            pass
        print(f"{pokemon.team}的{pokemon.name} 被麻痹了，无法行动！")
        
# 百分比减伤效果类
class DamageReductionEffect(Effect):
    name = "百分比减伤提升"

    def __init__(self, reduction_rate: float, duration: int = 1) -> None:
        super().__init__(duration)
        self.reduction_rate = reduction_rate  # 伤害减少比例

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用百分比减伤效果：为目标宝可梦添加伤害减少属性，提升减伤概率至100%
        pokemon.injury_reduction_rate = self.reduction_rate
        pokemon.Probability_of_damage_reduction = 1.0  # 提升减伤概率至100%

# 攻击力增强效果类
class AttackBoostEffect(Effect):
    name = "攻击力增强"

    def __init__(self, duration: int, level: int | None = None) -> None:
        # duration: 持续回合数；
        # level: 若显式传入（不为 None），表示希望叠加该层数；若不传入（None），视为仅刷新持续时间，不要叠加层数。
        super().__init__(duration, level)
        self.max_level = 4  # 设置攻击力增强效果的最大层数
        self.duration = 3
        # 每层给予的攻击力临时增量
        self.boost_per_level = 3

    def apply(self, pokemon: "Pokemon") -> None:
        # 临时攻击力提升
        pokemon.temporary_attack = self.boost_per_level * self.level
        print(f"{pokemon.team}的{pokemon.name} 暂时提升了 {pokemon.temporary_attack} 点攻击力！")

# 烧伤效果类
class BurnEffect(Effect):
    name = "烧伤" 

    def __init__(self, damage: int = 10, duration: int = 3) -> None:
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用烧伤效果：每回合对宝可梦造成固定伤害
        pre_hp = pokemon.hp
        pokemon.hp -= self.damage
        if pokemon.hp <= 0:
            pokemon.hp = max(0, pokemon.hp)
            try:
                pokemon.alive = False
            except Exception:
                pass

        print(f"{pokemon.team}的{pokemon.name} 受到 {pre_hp - pokemon.hp} 点烧伤伤害！剩余生命: {max(0, pokemon.hp)}/{pokemon.max_hp}")

# 冰冻效果类
class FreezeEffect(Effect):
    name = "冰冻"  

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用冰冻效果：将目标宝可梦在本回合设为无法行动
        try:
            pokemon.can_act = False
        except Exception:
            # 如果目标对象没有 can_act 属性，忽略
            pass
        print(f"{pokemon.team}的{pokemon.name} 被冰冻了，无法行动！")