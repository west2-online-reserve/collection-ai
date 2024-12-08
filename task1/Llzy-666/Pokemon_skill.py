from time import sleep
from typing import TYPE_CHECKING
import random
import Pokemon_statuses

if TYPE_CHECKING:
    from pokemon import Pokemon

class Skill():
    name: str
    skill_description: str

    #初始化技能类
    def __init__(self):
        pass
    #执行技能(接受参数)
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        raise NotImplementedError#异常处理
    #打印时返回技能名字 
    def __str__(self):
        return f"{self.name}"
    
#皮卡丘的技能
# 十万伏特
class ThunderBolt(Skill):
    name = "**十万伏特 (ThunderBolt)**"
    skill_description = "对敌人造成 1.4 倍攻击力的电属性伤害,并有 10% 概率使敌人麻痹."

    def __init__(self, damage:int, probability_of_paralysis:int = 10):
        super().__init__()
        self.damage = 1.4 * damage
        self.probability_of_paralysis = probability_of_paralysis
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
        print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")

        if random.randint(1, 100) <= self.probability_of_paralysis:
            if user.name.endswith("[电脑]"):
                duration = 2
            else:
                duration = 1
            opponent.add_statuses_effect(Pokemon_statuses.ParalysisEffect(duration = duration))
            print(f"十万伏特(ThunderBolt)麻痹了{opponent.name}")
        else:
            print(f"十万伏特(ThunderBolt)这次没能成功麻痹{opponent.name}.")

# 电光一闪
class QuickAttack(Skill):
    name = "**电光一闪 (Quick Attack)**"
    skill_description = "对敌人造成 1.0 倍攻击力的快速攻击(快速攻击有几率触发第二次攻击),10% 概率触发第二次"

    def __init__(self, damage:int, double_attack:int = 10):
        super().__init__()
        self.damage = damage
        self.double_attack = double_attack
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        if random.randint(1, 100) <= self.double_attack:
            print("触发了二次攻击!")
            first_damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
            second_damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
            damage_dealt = first_damage_dealt + second_damage_dealt
            print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")
        else:
            damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
            print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")

#妙蛙种子的技能
# 种子炸弹
class SeedBomb(Skill):
    name = "**种子炸弹 (SeedBomb):**"
    skill_description = "妙蛙种子发射一颗种子,爆炸后对敌方造成 1.0 倍攻击力的草属性伤害.若击中目标,目标有15%几率陷入“中毒”状态,每回合损失10生命值"

    def __init__(self, damage:int, poison_damage = 10, probability_of_poisoning:int = 15):
        super().__init__()
        self.damage = damage
        self.poison_damage = poison_damage
        self.probability_of_poisoning = probability_of_poisoning
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
        print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")

        if random.randint(1, 100) <= self.probability_of_poisoning:
            for status in opponent.statuses[:]:
                if status.name == "中毒":
                    status.duration = 3
                    print(f"种子炸弹重置了{opponent.name}的中毒时间")
            else:
                opponent.add_statuses_effect(Pokemon_statuses.PoisonEffect(self.poison_damage))
                print(f"种子炸弹使{opponent.name}中毒了!")
        else:
            print("种子炸弹这次并未使对方中毒.")
        
# 寄生种子
class ParasiticSeeds(Skill):
    name = "**寄生种子 (ParasiticSeeds):**"
    skill_description = "妙蛙种子向对手播种,每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合"

    def __init__(self, numeric_value:int = 10):
        super().__init__()
        self.numeric_value = numeric_value

    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        for status in user.statuses[:]:
            if status.name == "治疗":
                status.duration = 3
                print(f"寄生种子重置了{user.name}的治疗时间")
                break
        else:
            user.add_statuses_effect(Pokemon_statuses.HealEffect(self.numeric_value))
        for status in opponent.statuses[:]:
            if status.name == "寄生":
                status.duration = 3
                print(f"寄生种子重置了{opponent.name}的寄生时间")
                break
        else:
            opponent.add_statuses_effect(Pokemon_statuses.ParasitismEffect(self.numeric_value))
            print(f"寄生种子成功寄生了{opponent.name}!")
        
#杰尼龟的技能
# 水枪
class AquaJet(Skill):

    name = "**水枪 (AquaJet)**"
    skill_description = "杰尼龟喷射出一股强力的水流，对敌方造成 140% 水属性伤害"

    def __init__(self, damage:int):
        super().__init__()
        self.damage = 1.4 * damage
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
        print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")

# 护盾·
class Shield(Skill):
    name = "**护盾 (Shield)**"
    skill_description = "杰尼龟使用水流形成保护盾,减少下一回合受到的伤害50%"

    def __init__(self, numeric_value:int = 50):
        super().__init__()
        self.numeric_value = numeric_value

    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        super().__init__()
        if user.name.endswith("[电脑]"):
            duration = 2
        else:
            duration = 1
        user.add_statuses_effect(Pokemon_statuses.ShieldEffect(self.numeric_value, duration))
        print("杰尼龟为自己添加了护盾,下次受到的伤害可减免50%!")

#小火龙的技能
# 火花
class Ember(Skill):
    name = "**火花 (Ember)**"
    skill_description = "小火龙发射出一团小火焰,对敌人造成 100% 火属性伤害,并有10%的几率使目标陷入“烧伤”状态(每回合受到10额外伤害,持续2回合)"

    def __init__(self, damage:float, burn_damage = 10, probability_of_burn:int = 10):
        super().__init__()
        self.damage = damage
        self.burn_damage = burn_damage
        self.probability_of_burn = probability_of_burn
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
        print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")
        
        if random.randint(1, 100) <= self.probability_of_burn:
            for status in opponent.statuses[:]:
                if status.name == "烧伤":
                    status.duration = 2
                    print(f"火花重置了{opponent.name}的烧伤时间")
            else:
                opponent.add_statuses_effect(Pokemon_statuses.BurnEffect(self.burn_damage))
                print(f"火花使{opponent.name}烧伤了!")
        else:
            print("火花这次并未使对方烧伤.")
            
# 蓄能爆炎
class FlameCharge(Skill):
    name = "**蓄能爆炎 (Flame Charge)**"
    skill_description = "小火龙召唤出强大的火焰,对敌人造成 300% 火属性伤害,并有80%的几率使敌人陷入“烧伤”状态,这个技能需要1个回合的蓄力,并且在面对该技能时敌方闪避率增加 20%"
    
    def __init__(self, damage:float, burn_damage = 10, probability_of_burn:int = 80):
        super().__init__()
        self.damage = damage * 3
        self.burn_damage = burn_damage
        self.probability_of_burn = probability_of_burn

    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        if not user.is_charging: #判断是否充能完毕
            print(f"{user.name}当前正在蓄力...!")
            user.is_charging = True
            opponent.evasion_rate += 20
        elif user.is_charging:
            print(f"{user.name}的{self.name}已经蓄力完毕,即将对{opponent.name}造成攻击!")
            damage_dealt = opponent.damage_calculate(self.damage, user, opponent)
            print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")
            user.is_charging = False
            opponent.evasion_rate -=20

            if random.randint(1, 100) <= self.probability_of_burn:
                for status in opponent.statuses[:]:
                    if status.name == "烧伤":
                        status.duration = 2
                        print(f"火花重置了{opponent.name}的烧伤时间")
                else:
                    opponent.add_statuses_effect(Pokemon_statuses.BurnEffect(self.burn_damage))
                    print(f"火花使{opponent.name}烧伤了!")
            else:
                print("火花这次并未使对方烧伤.")

#哥哥的技能
# 唱跳Rap篮球,Music~
class Ctrl_music(Skill):
    name = "**唱跳Rap篮球,Music~ (Ctrl_music)**"
    skill_description = "坤坤迫不及待展示才艺,通过精湛运球为自己提供护盾(66.6%的减伤),并且提供恢复效果(每回合回复6点HP)"

    def __init__(self, numeric_value_of_shield: float = 66.6, numeric_value_of_heal: int = 6):
        super().__init__()
        self.numeric_value_of_shield = numeric_value_of_shield
        self.numeric_value_of_heal = numeric_value_of_heal
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        super().__init__()
        if user.name.endswith("[电脑]"):
            duration = 2
        else:
            duration = 1
        user.add_statuses_effect(Pokemon_statuses.ShieldEffect(self.numeric_value_of_shield, duration))
        print("坤坤为自己添加了护盾,下次受到的伤害可减免66.6%!")
        for status in user.statuses[:]:
            if status.name == "治疗":
                status.duration = 3
                print(f"唱跳Rap篮球,Music~重置了{user.name}的治疗时间")
                break
        else:
            user.add_statuses_effect(Pokemon_statuses.HealEffect(self.numeric_value_of_heal))
            print("坤坤为自己添加了恢复效果,每回合恢复6点HP!")

# 鸡你太美
class JNTM(Skill):
    name = "**鸡你太美 (JNTM)**"
    skill_description = "坤坤开启帝王舞姿,对敌方造成攻击力6倍的真实伤害(无视防御力)!并有几率(66.6%)使敌方为之Crush,陷入烧伤状态,并且麻痹!"

    def __init__(self, damage:int, probability_of_crush = 66.6, burn_damage = 6):
        super().__init__()
        self.damage = damage * 6
        self.burn_damage = burn_damage
        self.probability_of_crush = probability_of_crush
    
    def execute(self, user:"Pokemon", opponent:"Pokemon"):
        true_damage = self.damage + opponent.defence
        damage_dealt = opponent.damage_calculate(true_damage, user, opponent)
        print(f"{user.name}使用了{self.name},造成了{damage_dealt}点伤害给{opponent.name}")
        if random.randint(1, 100) <= self.probability_of_crush:
            if user.name.endswith("[电脑]"):
                duration = 2
            else:
                duration = 1
            opponent.add_statuses_effect(Pokemon_statuses.ParalysisEffect(duration))
            for status in opponent.statuses[:]:
                if status.name == "烧伤":
                    status.duration = 2
                    print(f"鸡你太美重置了{opponent.name}的烧伤时间")
            else:
                opponent.add_statuses_effect(Pokemon_statuses.BurnEffect(self.burn_damage))
            print(f"{opponent.name}Crush On You!")
            print(f"{opponent.name}烧伤并且麻痹了!")
        else:
            print(f"{opponent.name} doesn't crush you T-T.")