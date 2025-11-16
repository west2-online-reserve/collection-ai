# 导入所需模块
import random  # 用于随机数生成
from typing import TYPE_CHECKING  # 用于类型检查
import effects  # 导入效果模块

# 条件导入，避免循环导入问题
if TYPE_CHECKING:
    from pokemon import Pokemon

# 技能基类
class Skill:
    name: str  # 技能名称

    def __init__(self) -> None:
        """
        技能基类的初始化方法
        """
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        """
        技能执行的抽象方法
        param user: 使用技能的宝可梦
        param opponent: 技能的目标宝可梦
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        返回技能的字符串表示
        """
        return f"{self.name}"


# 种子炸弹技能类
class SeedBomb(Skill):
    name = "种子炸弹"  

    def __init__(self, damage: int, activation_chance: int = 15) -> None:
        """
        初始化种子炸弹技能
        :param damage: 基础伤害值
        :param activation_chance: 触发中毒效果的几率
        """
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 毒性效果触发概率

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行种子炸弹技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦
        """
        # 对目标造成基础伤害（传入攻击者以便触发被动效果，例如闪避回调）
        pre_hp = opponent.hp
        opponent.receive_damage(self.damage, user)
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}, 对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
        )

        # 随机判断是否触发中毒效果
        if random.randint(1, 100) <= self.activation_chance:
            # 触发成功，添加中毒状态
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 中毒了！")
        else:
            # 触发失败
            print(f"{user.team}的{self.name} 没有成功中毒 {opponent.team}的{opponent.name}。")


# 寄生种子技能类
class ParasiticSeeds(Skill):
    name = "寄生种子" 

    def __init__(self, amount: int) -> None:
        """
        初始化寄生种子技能
        :param amount: 每回合恢复的生命值数量
        """
        super().__init__()
        self.amount = amount

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行寄生种子技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦
        """
        # 为使用者添加治疗效果
        user.add_status_effect(effects.HealEffect(self.amount))
        print(f"————{user.team}的{user.name} 使用了 {self.name}，恢复了 {self.amount} HP————")

        # 为对手添加中毒效果
        opponent.add_status_effect(effects.PoisonEffect())
        print(f"————{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 中毒了！————")

# 十万伏特技能类
class Thunderbolt(Skill):
    name = "十万伏特"

    def __init__(self, damage: int, activation_chance: int = 10) -> None:
        """
        初始化电击技能
        :param damage: 基础伤害值
        :param activation_chance: 触发麻痹效果的几率（默认10%）
        """
        super().__init__()
        self.damage = damage * 1.4  # 对敌人造成 1.4 倍攻击力的电属性伤害
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行电击技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦
        """
        # 对目标造成基础伤害（传入攻击者以便触发被动效果，例如闪避回调）
        pre_hp = opponent.hp
        opponent.receive_damage(self.damage, user)
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}, 对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
        )

        # 随机判断是否触发麻痹效果
        if random.randint(1, 100) <= self.activation_chance:
            # 根据目标队伍决定麻痹生效时机：
            # - 电脑遭到麻痹：立即失去当前回合行动（玩家先手，可造成影响）
            # - 玩家遭到麻痹：标记下一回合跳过（当前回合已行动或还未进入其行动阶段）
            if getattr(opponent, 'team', None) == '玩家':
                opponent.next_turn_forced_skip = True
                print(f"————{opponent.team}的{opponent.name} 被 {self.name} 麻痹了！将跳过下一回合的行动。————")
            else:
                effects.ParalyzeEffect().apply(opponent)
                print(f"————{opponent.team}的{opponent.name} 被 {self.name} 麻痹了！本回合失去行动权。————")
        else:
            # 触发失败
            print(f"————{self.name} 没有成功麻痹 {opponent.team}的{opponent.name}。————")

# 电光一闪技能类
class QuickAttack(Skill):
    name = "电光一闪"

    def __init__(self, damage: int, continuous: int =10) -> None:
        """
        初始化快速攻击技能
        :param damage: 基础伤害值
        :param continuous: 触发第二次攻击的几率（默认10%）
        """
        super().__init__()
        self.damage = damage
        self.continuous = continuous

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行快速攻击技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦
        """
        # 对目标造成基础伤害
        pre_hp = opponent.hp
        opponent.receive_damage(self.damage, user)
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}，对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
        )

        # 随机判断是否触发第二次攻击
        if random.randint(1, 100) <= self.continuous:
            print(f"————{user.team}的{user.name} 触发了第二次攻击！————")
            pre_hp = opponent.hp
            opponent.receive_damage(self.damage, user)
            print(
                f"————对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
            )

# 水枪技能类
class Aqua_Jet(Skill):
    name = "水枪"

    def __init__(self, damage: int) -> None:
        """
        初始化水枪技能
        :param damage: 基础伤害值
        """
        super().__init__()
        self.damage = damage*1.4  # 对敌人造成 1.4 倍攻击力的水属性伤害

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行水枪技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦
        """
        # 对目标造成基础伤害
        pre_hp = opponent.hp
        opponent.receive_damage(self.damage, user)
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}，对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
        )

# 护盾技能类
class Shield(Skill):
    name = "护盾"

    def __init__ (self, shield_rate: float) -> None:
        """
        初始化护盾技能
        """
        super().__init__()
        self.shield_rate = shield_rate

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        """
        执行护盾技能
        :param user: 使用技能的宝可梦
        :param opponent: 技能目标宝可梦（护盾技能不使用该参数）
        """
        # 为使用者添加护盾效果
        user.add_status_effect(effects.DamageReductionEffect(reduction_rate=self.shield_rate))
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}，下回合获得 {self.shield_rate * 100}% 减伤的护盾效果！————"
        )

# 火花技能类
class Ember(Skill):
    name = "火花"

    def __init__(self, damage: int ,Pokemon: "Pokemon") -> None:
        super().__init__()
        # 基础伤害（通常等于使用者的基础攻击）
        self.damage = damage
        self.burn_chance = 10  # 默认10%烧伤几率

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 每次释放时按当前攻击+临时攻击计算
        temp_attack = getattr(user, 'temporary_attack', 0)
        real_damage = self.damage + temp_attack

        pre_hp = opponent.hp
        opponent.receive_damage(real_damage, user)
        #如果对方受到伤害，增加一层攻击力增强效果
        if opponent.hp < pre_hp:
            user.add_status_effect(effects.AttackBoostEffect(duration=3, level=1))
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}, 对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
        )

        # 随机判断是否触发烧伤效果
        if random.randint(1, 100) <= self.burn_chance:
            # 触发成功，添加烧伤状态
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 烧伤了！")
        else:
            # 触发失败
            print(f"{user.team}的{self.name} 没有成功烧伤 {opponent.team}的{opponent.name}。")

# 蓄能爆炎技能类
class Flame_Charge(Skill):
    name = "蓄能爆炎"

    def __init__(self, damage: int, Pokemon: "Pokemon") -> None:
        super().__init__()
        self.damage = damage
        self.burn_chance = 80  # 默认80%烧伤几率

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if getattr(user, 'Gathering_strength', False):
            #对方的闪避率临时增加20%
            opponent_pre_evade_chance = opponent.evade_chance
            opponent.evade_chance += 20

            # 蓄力释放时：按当前攻击+临时攻击的3倍计算
            temp_attack = getattr(user, 'temporary_attack', 0)
            real_damage = (self.damage + temp_attack) * 3

            pre_hp = opponent.hp
            opponent.receive_damage(real_damage, user)
            print(
                f"————{user.team}的{user.name} 使用了 {self.name}, 对 {opponent.team}的{opponent.name} 造成了 {pre_hp - opponent.hp} 点伤害————"
            )
            # 恢复对方的闪避率
            opponent.evade_chance = opponent_pre_evade_chance

            # 随机判断是否触发烧伤效果
            if random.randint(1, 100) <= self.burn_chance:
                # 触发成功，添加烧伤状态
                opponent.add_status_effect(effects.BurnEffect())
                print(f"{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 烧伤了！")
            else:
                # 触发失败
                print(f"{user.team}的{self.name} 没有成功烧伤 {opponent.team}的{opponent.name}。")
            user.Gathering_strength = False
        else:
            user.Gathering_strength = True
            print(f"————{user.team}的{user.name} 使用了 {self.name}，正在蓄力中！下次使用将造成巨大伤害！————")


# 冰冻盾技能类
class Blizzard(Skill):
    name = "急冻盾"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        # 为使用者添加护盾效果
        user.add_status_effect(effects.DamageReductionEffect(reduction_rate=0.5))
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}，下回合获得 50% 减伤的护盾效果！————"
        )

# 冰冻光束技能类
class Ice_Beam(Skill):
    name = "冰冻光束"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        #  receive_damage 统一处理防御与闪避
        pre_hp = opponent.hp
        opponent.receive_damage(self.damage, user)
        actual = pre_hp - opponent.hp
        print(
            f"————{user.team}的{user.name} 使用了 {self.name}, 对 {opponent.team}的{opponent.name} 造成了 {actual} 点伤害————"
        )

        # 随机判断是否触发冰冻效果
        if random.randint(1, 100) <= 40:  # 40%几率触发冰冻
            # 根据目标队伍决定冰冻生效时机：
            # - 电脑被冻结：立即失去当前回合行动
            # - 玩家被冻结：跳过下一回合行动
            if getattr(opponent, 'team', None) == '玩家':
                opponent.next_turn_forced_skip = True
                print(f"{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 冰冻了！将跳过下一回合的行动。")
            else:
                effects.FreezeEffect().apply(opponent)
                print(f"{opponent.team}的{opponent.name} 被 {user.team}的{user.name} 使用 {self.name} 冰冻了！本回合失去行动权。")
        else:
            print(f"{user.team}的{self.name} 没有成功冰冻 {opponent.team}的{opponent.name}。")