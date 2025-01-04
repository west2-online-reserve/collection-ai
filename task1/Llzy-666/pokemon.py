from time import sleep
import random
import Pokemon_skill
from Pokemon_skill import Skill
from Pokemon_statuses import Effect

# 判断输入的选择是否合法
def valid_choice(choice, range):
    return choice.isdigit() and 1 <= int(choice) <= range

#宝可梦类
class Pokemon:
    name:str
    attributes:str

#初始化宝可梦所具有的属性
    def __init__(self, hp:int, attack:int, defence:int, evasion_rate:float):
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defence = defence
        self.skills = self.initialize_skill()
        self.evasion_rate = evasion_rate
        self.alive = True
        self.statuses = []

#抽象方法，子类应实现具体技能的初始化
    def initialize_skill(self):
        raise NotImplementedError
    
#打印对象时返回以下信息    
    def __str__(self):
        return f"**{self.name}**(属性:{self.attributes})"

#使用宝可梦技能
    def use_skill(self, skill:Skill, opponent:"Pokemon"):
        print(f"{self.name}使用了{skill.name}")
        print()
        sleep(1)
        skill.execute(self, opponent)

#属性克制
    def attribute_restraint(self, opponent:"Pokemon"):
        magnification = 1.0
        self_attribute = self.attributes
        opponent_attribute = opponent.attributes
        # 草克制水,火克制草,水克制火,电克制水
        if (self_attribute == "Grass" and opponent_attribute == "Water") or \
           (self_attribute == "Fire" and opponent_attribute == "Grass") or \
           (self_attribute == "Water" and opponent_attribute == "Fire") or \
           (self_attribute == "Electrical" and opponent_attribute == "Water"):
            magnification = 2.0
        # 草被火克制,火被水克制,水被电克制,电被草克制
        elif (self_attribute == "Grass" and opponent_attribute == "Fire") or \
            (self_attribute == "Fire" and opponent_attribute == "Water") or \
            (self_attribute == "Water" and opponent_attribute == "Electrical") or \
            (self_attribute == "Electrical" and opponent_attribute == "Grass"):
            magnification = 0.5
        elif self_attribute == "Kun" or opponent_attribute == "Kun": #坤无克制
            magnification = 1.0
        return magnification

#判断有无属性克制
    def judge_attribute_restraint(self, opponent:"Pokemon"):
        return self.attribute_restraint(opponent)

#添加状态效果
    def add_statuses_effect(self, effect:Effect):
        self.statuses.append(effect)

#应用当前的状态效果
    def apply_statuses_effect(self):
        for status in self.statuses[:]: #使用切片防止列表在遍历时被修改
            status.apply(self) #运用
            status.decrease_duration() #减少持续时间
            if status.duration <= 0:
                print(f"{self.name}的{status.name}效果被移除了.")
                self.statuses.remove(status) #移除状态效果
                print()

#计算造成的伤害并返回受到的伤害及剩余的HP
    def damage_calculate(self, damage, attacker:"Pokemon", opponent:"Pokemon"):
        if random.randint(1, 100) <= self.evasion_rate:
            damage = 0
            print(f"{self.name}成功闪避了这次攻击")
            #如果成功闪避,不消耗护盾次数
            for status in self.statuses:
                if status.name == "护盾":
                    status.duration += 1

            if self.name == "PikaChu": #检测皮卡丘
                self.counterattack(attacker) #反击给攻击者
            elif self.name == "PikaChu[电脑]": #检测电脑皮卡丘
                self.computer_counterattack(attacker) 
            return damage # 不再进行后续代码
        
        damage -= self.defence
        restraint_factor = attacker.judge_attribute_restraint(opponent)
        damage *= restraint_factor

        if restraint_factor != 1.0:
            print(f"存在属性克制!{attacker.name}将造成伤害倍率为{restraint_factor}的伤害")
        
        #判断水属性宝可梦减伤
        if self.attributes == "Water":
            random.randint(1, 100) <= 50
            damage *= 0.7 #减伤百分之三十
            print(f"{self.name}触发了被动,本次受到的伤害减少30%")

        #判断是否护盾减伤
        for status in self.statuses:
            if status.name == "护盾":
                damage *= (status.numeric_value / 100)
                print(f"{self.name}的护盾成功生效了!本次受到伤害减少{status.numeric_value}%!")

        if damage <= 0:
            print(f"{self.name}的防御力完全吸收了来自{attacker.name}的伤害")
            damage = 0
            return damage # 不再进行后续代码
        damage = round(damage, 2) # 保留两位小数,美观
        #计算剩余生命值
        self.hp -= damage
        self.hp = round(self.hp, 2)
        print(f"{self.name}从{attacker.name}的攻击中受到了{damage}点伤害! 剩余 HP: {self.hp}/{self.max_hp}")
        #判断是否死亡
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name}昏厥了!")
        return damage

#恢复HP
    def heal_self(self, numeric_value):
            # 为自身恢复生命值
            if not isinstance(numeric_value, int):
                numeric_value = int(numeric_value)

            self.hp += numeric_value
            if self.hp > self.max_hp:
                self.hp = self.max_hp
            print(f"{self.name}治疗了{numeric_value}点HP!现在的血量为{self.hp}/{self.max_hp}HP.")

#新回合开始
    def begin(self):
        self.apply_statuses_effect()

#雷属性宝可梦类
class ElectricalPokemon(Pokemon):
    attributes = "Electrical"
    def __init__(self, hp, attack, defence, evasion_rate):
        super().__init__(hp, attack, defence, evasion_rate)
    
    def begin(self):
        pass

#皮卡丘
class PikaChu(ElectricalPokemon):
    name = "PikaChu"
    def __init__(self, hp = 80, attack = 35, defence = 5, evasion_rate = 30):
        super().__init__(hp, attack, defence, evasion_rate)
    
    def initialize_skill(self):
        return [Pokemon_skill.ThunderBolt(damage = self.attack), Pokemon_skill.QuickAttack(damage = self.attack)]

#玩家皮卡丘闪避成功后反击
    def counterattack(self, opponent:"Pokemon"):
        print(f"请选择{self.name}的技能进行反击：")
        for idx, skill in enumerate(self.skills):
            print(f"{idx + 1}. {skill.name}")
        choice = input("请输入对应的数字来选择一个你要使用的技能: ")
        if valid_choice(choice, len(self.skills)):
            player_skill = self.skills[int(choice) - 1]
            self.use_skill(player_skill, opponent)
        else:
            print("请输入有效的数字来选取技能!")

#电脑皮卡丘闪避成功后反击
    def computer_counterattack(self, opponent:"Pokemon"):
        print(f"{self.name}随机选择了技能进行反击：")
        if self.skills:  # 确保技能列表不为空
            chosen_skill = random.choice(self.skills)
            print(f"{self.name}使用了{chosen_skill.name}对{opponent.name}")
            chosen_skill.execute(self, opponent)

#草属性宝可梦类
class GrassPokemon(Pokemon):
    attributes = "Grass"
    def __init__(self, hp, attack, defence, evasion_rate):
        super().__init__(hp, attack, defence, evasion_rate)

#草属性被动
    def grass_attribute(self):
        #草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name}治疗了{amount}点HP.现在的血量为:{self.hp}/{self.max_hp}HP.")

#妙蛙种子
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"
    def __init__(self, hp = 100, attack = 35, defence = 10, evasion_rate = 10):
        super().__init__(hp, attack, defence, evasion_rate)
    
    def initialize_skill(self):
        return [Pokemon_skill.SeedBomb(damage = self.attack), Pokemon_skill.ParasiticSeeds(numeric_value = 10)]
    
#水属性宝可梦类
class WaterPokemon(Pokemon):
    attributes = "Water"
    def __init__(self, hp, attack, defence, evasion_rate):
        super().__init__(hp, attack, defence, evasion_rate)

#杰尼龟
class Squirtle(WaterPokemon):
    name = "Squirtle"
    def __init__(self, hp = 80, attack = 25, defence = 20, evasion_rate = 20):
        super().__init__(hp, attack, defence, evasion_rate)

    def initialize_skill(self):
        return [Pokemon_skill.AquaJet(damage = self.attack), Pokemon_skill.Shield(numeric_value = 50)]
    
#火属性宝可梦类
class FirePokemon(Pokemon):
    attributes = "Fire"
    def __init__(self, hp, attack, defence, evasion_rate):
       super().__init__(hp, attack, defence, evasion_rate)
       self.attack_level = 0
    
    def begin(self):
        self.fire_attribute()
        self.skills = self.initialize_skill()
    
    def fire_attribute(self):# 每回合增加10%攻击力，最多增加4次
        amount = 10
        if self.attack_level < 4:
            self.attack_level += 1
            self.attack *= (1 + (amount / 100)) # 攻击力增加10%
            self.attack = round(self.attack, 2)
            print(f"{self.name}的攻击提高了,现在的攻击力为{self.attack}点.")
        else:
            print(f"{self.name}的被动已叠加四层,现在的攻击力为{self.attack}点.")
        
#小火龙
class Charmander(FirePokemon):
    name = "Charmander"
    def __init__(self, hp = 80, attack = 35, defence = 15, evasion_rate = 10):
        super().__init__(hp, attack, defence, evasion_rate)
        self.is_charging = False

    def initialize_skill(self):
        self.skills = [Pokemon_skill.Ember(damage = self.attack), Pokemon_skill.FlameCharge(damage = self.attack)]
        return self.skills
    
#定义坤属性宝可梦类
class DarkPokemon(Pokemon):
    attributes = "Kun"
    def __init__(self, hp, attack, defence, evasion_rate):
        super().__init__(hp, attack, defence, evasion_rate)

#真爱坤(树脂666)
class Kunkun(DarkPokemon):
    name = "Kunkun"
    def __init__(self, hp = 66, attack = 6, defence = 6.6, evasion_rate = 6.66):
        super().__init__(hp, attack, defence, evasion_rate)

    def initialize_skill(self):
        return [Pokemon_skill.Ctrl_music(numeric_value_of_shield = 66.6, numeric_value_of_heal = 6.6), Pokemon_skill.JNTM(damage = self.attack)]