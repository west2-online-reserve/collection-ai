from __future__ import annotations
import random
import skills
from skills import Skill  
import effects  
from effects import Effect  


class Pokemon:
    name: str  # 宝可梦名称
    type: str  # 宝可梦属性类型

    def __init__(self, hp: int, attack: int, defense: int, evade_chance: int) -> None:
        self.hp = hp  # 当前生命值
        self.max_hp = hp  # 最大生命值
        self.attack = attack  # 攻击力
        self.temporary_attack = 0  # 临时攻击力
        self.defense = defense  # 防御力
        self.evade_chance = evade_chance # 每回合或每次攻击时的闪避几率（1-100），默认为0
        self.can_act = True  # 当前回合是否可以行动
        self.skills = self.initialize_skills()  # 初始化技能列表
        self.alive = True  # 存活状态
        self.team = None  # 所属队伍
        self.statuses = []  # 状态效果列表
        

    def initialize_skills(self):

        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        """
        使用技能攻击对手
        """
        print(f"{self.team}的{self.name} 使用了 {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        """
        治疗自身生命值
        :param amount: 要恢复的生命值数量
        """
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp 
        print(f"{self.team}的{self.name} 治疗了 {amount} HP! 当前 HP: {self.hp}/{self.max_hp}")

    def receive_damage(self, damage, attacker=None):
        """
        承受伤害，考虑闪避、计算实际伤害值并更新生命值
        :param damage: 受到的原始伤害值
        :param attacker: 发起攻击的宝可梦（
        """
        if not isinstance(damage, int):
            damage = int(damage)

        # 先判断是否闪避命中（闪避成功则不会受到伤害，并触发 on_evade）
        if hasattr(self, "evade_chance") and self.evade_chance > 0:
            roll = random.randint(1, 100)
            if roll <= self.evade_chance:
                print(f"{self.team}的{self.name} 闪避了攻击！")
                try:
                    self.on_evade(attacker)
                except Exception:
                    # 确保闪避回调不会中断战斗流程
                    pass
                return

        # 计算实际伤害（原始伤害减去防御力）
        damage -= self.defense
        if damage <= 0:
            print(f"{self.team}的{self.name} 的防御吸收了攻击！")
            return

        self.hp -= damage
        print(
            f"{self.team}的{self.name} 受到了 {damage} 点伤害！剩余 HP: {max(0, self.hp)}/{self.max_hp}"
        )
        # 检查是否失去战斗能力
        if self.hp <= 0:
            self.alive = False
            print(f"{self.team}的{self.name} 已经失去战斗能力！")

    def on_evade(self, attacker: Pokemon):
        """
        被动回调：当成功闪避一次攻击时触发。
        默认不做任何事情，子类（例如电属性）可以覆盖以实现反击或特殊效果。
        :param attacker: 发起攻击的宝可梦对象
        """
        return

    def add_status_effect(self, effect: Effect):
        """
        添加状态效果。
        - 如果已有相同类型的状态：
          - 当传入的 effect 实例未显式携带要叠加的层数，仅将持续时间更新为两者最大值，不重复添加。
          - 当传入的 effect 实例携带了层数（即 effect.level 为整数且大于 1），则持续时间更新为两者最大值，且将传入的层数叠加到已有状态的层数上（叠加后会按照已有状态的 max_level 或默认 4 上限裁剪）。
        - 若没有相同类型的状态，则直接添加到状态列表中。
        当传入的 effect.level > 1 时认为调用者希望叠加层数；否则视为不带层数，仅刷新持续时间。

        :param effect: 要添加的状态效果
        """
        for existing in self.statuses:
            if type(existing) is type(effect):
                # 先刷新持续时间为两者最大值（如果两个实例都有 duration）
                if hasattr(existing, "duration") and hasattr(effect, "duration"):
                    existing.duration = max(existing.duration, effect.duration)

                # 判断传入实例是否显式携带了层数（现在由 Effect 在构造时设置 _explicit_level）
                add_level = 0
                try:
                    if getattr(effect, "_explicit_level", False):
                        # 显式携带层数（即使是 level==1 也算显式），把传入的层数作为要叠加的值
                        if hasattr(effect, "level") and isinstance(effect.level, int):
                            add_level = int(effect.level)
                except Exception:
                    add_level = 0

                if add_level > 0:
                    # 叠加层数到 existing（若 existing 有 max_level 属性则使用它作为上限；若无则默认无上限）
                    try:
                        cur = int(getattr(existing, "level", 1))
                    except Exception:
                        cur = 1
                    try:
                        maxl = getattr(existing, "max_level", None)
                    except Exception:
                        maxl = None

                    new_level = cur + add_level
                    
                    if maxl is not None:
                        try:
                            maxl_int = int(maxl)
                        except Exception:
                            maxl_int = None
                        if maxl_int is not None and new_level > maxl_int:
                            new_level = maxl_int

                    try:
                        existing.level = new_level
                    except Exception:
                        # 如果不能写入 level，忽略层数修改
                        pass

                # 不重复添加（无论是否叠加层数），直接返回
                return

        # 没有同类型状态才添加
        self.statuses.append(effect)

    def apply_status_effect(self):
        """
        应用所有当前的状态效果，并移除已经结束的效果
        """
        for status in self.statuses[:]:
            try:
                status.apply(self)
            except Exception:
                pass
            try:
                status.decrease_duration(self)
            except Exception:
                pass
            if getattr(status, "duration", 0) <= 0:
                print(f"{self.team}的{self.name} 的 {status.name} 效果已经结束。")
                try:
                    self.statuses.remove(status)
                except ValueError:
                    pass

    def type_effectiveness(self, opponent: Pokemon):
        """
        计算属性克制关系，返回伤害倍率
        """
        raise NotImplementedError

    def begin(self):
        """
        回合开始时的处理方法
        """
        pass

    def __str__(self) -> str:
        """
        返回宝可梦的字符串表示
        """
        return f"{self.type}属性宝可梦：{self.name}"



# 草属性宝可梦类
class GrassPokemon(Pokemon):
    type = "草" 
    def type_effectiveness(self, opponent: Pokemon):
        """
        计算草属性对其他属性的克制关系
        :param opponent: 对手宝可梦
        :return: 伤害倍率
        """
        effectiveness = 1.0  # 默认伤害倍率
        opponent_type = opponent.type

        # 草属性克制关系：
        # 对水属性双倍伤害
        # 对火属性伤害减半
        if opponent_type == "水":
            effectiveness = 2.0
        elif opponent_type == "火":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        """
        回合开始时触发草属性特性
        """
        self.grass_attribute()

    def grass_attribute(self):
        """
        草属性特性：在每回合开始时恢复最大生命值的10%
        """
        amount = self.max_hp * 0.1  # 计算恢复量
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp  # 确保不超过最大生命值
        print(
            f"{self.team}的{self.name} 触发被动，恢复了 {amount} HP！当前 HP: {self.hp}/{self.max_hp}"
        )

class ElectricPokemon(Pokemon):
    type = "电" 

    def type_effectiveness(self, opponent: Pokemon):
        """
        计算电属性对其他属性的克制关系
        :param opponent: 对手宝可梦
        :return: 伤害倍率
        """
        effectiveness = 1.0  # 默认伤害倍率
        opponent_type = opponent.type

        # 电属性克制关系：
        # 对水属性双倍伤害
        # 对草属性伤害减半
        if opponent_type == "水":
            effectiveness = 2.0
        elif opponent_type == "草":
            effectiveness = 0.5
        return effectiveness
    
    def begin(self):
        """
        回合开始时触发电属性特性
        """
        self.electric_attribute()

    def electric_attribute(self):
        """
        电属性的回合开始处理（当前未定义额外开场效果）
        """
        return

    def on_evade(self, attacker: Pokemon):
        """
        电属性被动：成功闪避一次攻击后，可以立即使用一次技能。
        行为规则：
        - 仅在 attacker 不为 None 且自身有技能时触发。
        - 如果该宝可梦属于玩家（team == "玩家"），则由玩家交互选择技能。
        - 如果该宝可梦属于电脑或非玩家，则随机选择一个技能并使用。
        """
        # 基本前置检查
        if attacker is None:
            return
        if not self.skills:
            return

        # 若属于玩家，由玩家选择技能（交互）；否则由电脑随机选择
        chosen_skill = None
        try:
            if getattr(self, "team", None) == "玩家":
                # 玩家交互选择（在非交互环境中会触发异常并回退）
                print(f"{self.team}的{self.name} 闪避了攻击！请选择一个技能对 {attacker.name} 使用:")
                for i, s in enumerate(self.skills, 1):
                    print(f"{i}: {s}")

                while True:
                    choice = input("选择技能编号进行使用（直接回车使用第一个技能）:")
                    if choice == "":
                        chosen_skill = self.skills[0]
                        break
                    if choice.isdigit() and 1 <= int(choice) <= len(self.skills):
                        chosen_skill = self.skills[int(choice) - 1]
                        break
                    print("无效选择，请重试。")
            else:
                # 电脑随机选择一个技能

                chosen_skill = random.choice(self.skills)
                print(f"{self.team}的{self.name} 闪避了攻击，随机选择使用技能 {chosen_skill.name} 对 {attacker.name}。")
        except Exception:
            # 非交互环境或出错时回退为第一个技能
            chosen_skill = self.skills[0]

        # 执行选中的技能（对攻击者）
        try:
            self.use_skill(chosen_skill, attacker)
        except Exception:
            # 确保被动触发不会中断主流程
            pass

class WaterPokemon(Pokemon):
    type = "水"  

    def type_effectiveness(self, opponent: Pokemon):
        """
        计算水属性对其他属性的克制关系

        """
        effectiveness = 1.0  # 默认伤害倍率
        opponent_type = opponent.type

        # 水属性克制关系：
        # 对火属性双倍伤害
        # 对电属性伤害减半
        if opponent_type == "火":
            effectiveness = 2.0
        elif opponent_type == "电":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        """
        回合开始时触发水属性特性
        """
        self.water_attribute()
    def water_attribute(self):
        """
        水属性的回合开始处理
        """
        return
    
    def receive_damage(self, damage, attacker=None):
        """
        承受伤害，考虑闪避、计算实际伤害值并更新生命值
        """
        if not isinstance(damage, int):
            damage = int(damage)

        # 先判断是否闪避命中（闪避成功则不会受到伤害，并触发 on_evade）
        if hasattr(self, "evade_chance") and self.evade_chance > 0:
            roll = random.randint(1, 100)
            if roll <= self.evade_chance:
                print(f"{self.team}的{self.name} 闪避了攻击！")
                return

        # 计算实际伤害（原始伤害减去防御力）
        damage -= self.defense

        # 如果在减防之后伤害已经小于等于0，直接视为被防御吸收，返回
        if damage <= 0:
            print(f"{self.team}的{self.name} 的防御吸收了攻击！")
            return

        injury_reduction_rate = 0.3  # 伤害减少比率
        Probability_of_damage_reduction = 0.5  # 伤害减少概率
    
        # 随机判断是否触发伤害减少效果
        if random.random() < Probability_of_damage_reduction:
            damage = int(damage * (1 - injury_reduction_rate))
            print(f"{self.team}的{self.name} 触发了伤害减少效果！伤害减少至 {damage} 点。")
        if damage <= 0:
            print(f"{self.team}的{self.name} 的防御吸收了攻击！")
            return

        self.hp -= damage
        print(
            f"{self.team}的{self.name} 受到了 {damage} 点伤害！剩余 HP: {max(0, self.hp)}/{self.max_hp}"
        )
        # 检查是否失去战斗能力
        if self.hp <= 0:
            self.alive = False
            print(f"{self.team}的{self.name} 已经失去战斗能力！")
    
class FirePokemon(Pokemon):
    type = "火"  # 设定属性为"火"
    
    def type_effectiveness(self, opponent: Pokemon):
        """
        计算火属性对其他属性的克制关系
        :param opponent: 对手宝可梦
        :return: 伤害倍率
        """
        effectiveness = 1.0  # 默认伤害倍率
        opponent_type = opponent.type

        # 火属性克制关系：
        # 对草属性双倍伤害
        # 对水属性伤害减半
        if opponent_type == "草":
            effectiveness = 2.0
        elif opponent_type == "水":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        """
        回合开始时触发火属性特性
        """
        self.fire_attribute()

    def fire_attribute(self):
        return
                
class IcePokemon(Pokemon):
    type = "冰"  # 设定属性为"冰"

    def type_effectiveness(self, opponent: Pokemon):
        """
        计算冰属性对其他属性的克制关系
        :param opponent: 对手宝可梦
        :return: 伤害倍率
        """
        effectiveness = 1.0  # 默认伤害倍率
        opponent_type = opponent.type

        # 冰属性克制关系：
        # 对火属性伤害减半
        if opponent_type == "火":
            effectiveness = 0.5
        return effectiveness
        # 对水属性双倍伤害
        if opponent_type == "水":
            effectiveness = 2.0 
        return effectiveness

    def begin(self):
        """
        回合开始时触发冰属性特性
        """
        self.ice_attribute()

    def ice_attribute(self):
        return


# 妙蛙种子类，继承自草属性宝可梦
class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"  

    def __init__(self, hp=100, attack=50, defense=10, evade_chance=10) -> None:
        """
        初始化妙蛙种子的属性
        :param hp: 生命值，默认100
        :param attack: 攻击力，默认50
        :param defense: 防御力，默认10
        :param evade_chance: 闪避几率，默认10%
        """
        super().__init__(hp, attack, defense, evade_chance)

    def initialize_skills(self):
        """
        初始化妙蛙种子的技能列表
        :return: 包含种子炸弹和寄生种子两个技能的列表
        """
        return [
            skills.SeedBomb(damage=50),  # 种子炸弹，基础伤害50
            skills.ParasiticSeeds(amount=10)  # 寄生种子，每回合恢复10点生命值
        ]

class Pikachu(ElectricPokemon):
    name = "皮卡丘"  

    def __init__(self, hp=80, attack=35, defense=5, evade_chance=30) -> None:
        """
        初始化皮卡丘的属性
        :param hp: 生命值，默认80
        :param attack: 攻击力，默认35
        :param defense: 防御力，默认5
        :param evade_chance: 闪避几率，默认30%
        """
        super().__init__(hp, attack, defense,evade_chance)

    def initialize_skills(self):
        """
        初始化皮卡丘的技能列表
        :return: 包含电击和十万伏特两个技能的列表
        """
        return [
            skills.Thunderbolt(damage=self.attack),  # 十万伏特，基础伤害等于攻击力
            skills.QuickAttack(damage=self.attack)  # 电光一闪，基础伤害等于攻击力
        ]
    
class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self, hp=80, attack=25, defense=20, evade_chance=20) -> None:
        """
        初始化杰尼龟的属性
        :param hp: 生命值，默认80
        :param attack: 攻击力，默认25
        :param defense: 防御力，默认20
        :param evade_chance: 闪避几率，默认20%
        """
        super().__init__(hp, attack, defense, evade_chance)

    def initialize_skills(self):
        """
        初始化杰尼龟的技能列表
        :return: 包含水枪和泡沫两个技能的列表
        """
        return [
            skills.Aqua_Jet(damage=self.attack),  # 水流喷射，基础伤害等于攻击力
            skills.Shield(shield_rate=0.5)  # 护盾减伤率50%
        ]
    
    # 杰尼龟的伤害处理
    def receive_damage(self, damage, attacker=None):
        """
        承受伤害，考虑闪避、计算实际伤害值并更新生命值
        :param damage: 受到的原始伤害值
        :param attacker: 发起攻击的宝可梦（可选，用于触发被动能力）
        """
        if not isinstance(damage, int):
            damage = int(damage)

        # 先判断是否闪避命中（闪避成功则不会受到伤害，并触发 on_evade）
        if hasattr(self, "evade_chance") and self.evade_chance > 0:
            roll = random.randint(1, 100)
            if roll <= self.evade_chance:
                print(f"{self.team}的{self.name} 闪避了攻击！")
                return

        # 计算实际伤害（原始伤害减去防御力）
        damage -= self.defense
        
        # 如果在减防之后伤害已经小于等于0，直接视为被防御吸收
        if damage <= 0:
            print(f"{self.team}的{self.name} 的防御吸收了攻击！")
            return
        
        injury_reduction_rate = 0.3  # 伤害减少比率
        Probability_of_damage_reduction = 0.5  # 伤害减少概率

        # 判断是否有减伤效果（DamageReductionEffect）并应用
        for status in self.statuses:
            # 状态效果位于 effects 模块，检查 DamageReductionEffect 并使用其 reduction_rate
            if isinstance(status, effects.DamageReductionEffect):
                damage = int(damage * (1 - status.reduction_rate))
                print(f"{self.team}的{self.name} 的护盾效果生效！伤害减少至 {damage} 点。")
                break
    
        # 随机判断是否触发伤害减少效果
        if random.random() < Probability_of_damage_reduction:
            damage = int(damage * (1 - injury_reduction_rate))
            print(f"{self.team}的{self.name} 触发了伤害减少效果！伤害减少至 {damage} 点。")
        if damage <= 0:
            print(f"{self.team}的{self.name} 的防御吸收了攻击！")
            return

        self.hp -= damage
        print(
            f"{self.team}的{self.name} 受到了 {damage} 点伤害！剩余 HP: {max(0, self.hp)}/{self.max_hp}"
        )
        # 检查是否失去战斗能力
        if self.hp <= 0:
            self.alive = False
            print(f"{self.team}的{self.name} 已经失去战斗能力！")

class Charmander(FirePokemon):
    name = "小火龙"
    def __init__(self, hp=80, attack=35, defense=15, evade_chance=10) -> None:
        """
        初始化小火龙的属性
        :param hp: 生命值，默认80
        :param attack: 攻击力，默认35
        :param defense: 防御力，默认15
        :param evade_chance: 闪避几率，默认10%
        """
        super().__init__(hp, attack, defense, evade_chance)
        self.Gathering_strength=False  # 蓄力状态标志

    def initialize_skills(self):
        """
        初始化小火龙的技能列表
        :return: 包含喷火和火焰旋涡两个技能的列表
        """
        return [
            skills.Ember(damage=self.attack, Pokemon=self),  # 火花，技能基础伤害等于攻击力
            skills.Flame_Charge(damage=self.attack, Pokemon=self)  # 蓄能爆炎，技能基础伤害等于攻击力
        ]

class Elegant_penguin(IcePokemon):
    name = "高雅企鹅"

    def __init__(self, hp=80, attack=30, defense=25, evade_chance=15) -> None:
        """
        初始化高雅企鹅的属性
        :param hp: 生命值，默认80
        :param attack: 攻击力，默认30
        :param defense: 防御力，默认25
        :param evade_chance: 闪避几率，默认15%
        """
        super().__init__(hp, attack, defense, evade_chance)

    def initialize_skills(self):
        """
        初始化高雅企鹅的技能列表
        :return: 包含冰冻光束和急冻光线两个技能的列表
        """
        return [
            skills.Ice_Beam(damage=self.attack),  # 冰冻光束，技能基础伤害等于攻击力
            skills.Blizzard(damage=self.attack)  # 急冻盾，护盾效果
        ]