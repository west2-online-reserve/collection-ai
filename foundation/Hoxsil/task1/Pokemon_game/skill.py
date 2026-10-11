import random

import constants
from pokemon import Pokemon

# 属性判断，克制返回 2，被克制返回 0.5，无影响返回 1
def attribute_judgement(self_attribute:str, oppnent_attribute:str) -> float:
    if self_attribute == "none" or oppnent_attribute == "none":
        return 1
    else:
        return constants.attribute_restraint[self_attribute, oppnent_attribute]

# num% 的几率返回 True
def odds_produce(num:int) -> bool:
    return random.randint(1,100) <= num

# 技能释放打印
def skill_use_print(self:Pokemon, skill_name:str) -> None:
    print(f"{self.name} 使用了 {skill_name}!")

# opponent 扣血并打印
def take_damage(self:Pokemon, opponent:Pokemon, damage:float, name:str = "none") -> None:
    # 水被动生效
    if "水被动" in opponent.buff:
        if odds_produce(constants.WATER_PASSIVE_REDUCE_CHANCE):
            print(f"{opponent.name} 的 水被动 减免了本次伤害！")
            damage *= 1 - constants.WATER_PASSIVE_REDUCE_RATIO
    # 火被动生效
    if "火被动" in self.buff:
        if not "火被动攻击力" in self.buff:
            self.buff["火被动攻击力"] = -1
        else:
            self.buff["火被动攻击力"] -= 1
            if self.buff["火被动攻击力"] < -4:
                self.buff["火被动攻击力"] = -4
    
    opponent.hp -= damage
    if name == "none":
        print(f"{opponent.name} 受到了 {damage:.3f} 点伤害！剩余 HP：{opponent.hp:.3f}")
    else:
        print(f"{opponent.name} 受到了 {name} 的 {damage:.3f} 点伤害！剩余 HP：{opponent.hp:.3f}")

# self 治疗并打印
def take_cure(self:Pokemon, opponent:Pokemon, cure:float, name:str = "none") -> None:
    self.hp += cure
    if self.hp > self.max_hp:
        self.hp = self.max_hp
    if name == "none":
        print(f"{self.name} 受到了 {cure} 点治疗！剩余 HP：{self.hp}")
    else:
        print(f"{self.name} 受到了 {name} 的 {cure} 点治疗！剩余 HP：{self.hp}")

# 计算伤害
def skill_damage_calculate(self:Pokemon, opponent:Pokemon, ratio:float, attribute:str) -> float:
    # 技能面板伤害
    base_damage:float = ratio * (self.attack + self.tmp_attack)
    # 技能元素加成伤害
    attribute_damage:float = base_damage * attribute_judgement(attribute, opponent.attribute)
    # 技能增幅/削减后伤害再减去对方防御值
    real_damage:float = attribute_damage * self.tmp_increase * opponent.tmp_reduce - (opponent.defense + opponent.tmp_defence)
    if real_damage < 0:
        real_damage = 0
    if "护盾" in opponent.buff:
        real_damage *= 1 - constants.SHIELD_REDUCE_RATIO
        if opponent.buff["护盾"] == 1:
            opponent.buff.pop("护盾")
        else:
            opponent.buff["护盾"] -= 1
    return real_damage

# 增加 buff
def buff_add(pokemon:Pokemon, name:str, num:int) -> None:
    if name in pokemon.buff:
        pokemon.buff[name] += num
        print(f"{pokemon.name} 的 {name} 层数增加了 {num} 层，现为 {pokemon.buff[name]} 层")
    else:
        pokemon.buff[name] = num
        print(f"{pokemon.name} 被施加了 {num} 层的 {name}")

# 判断闪避，命中 True，未命中 False
def dod_judgement(opponent:Pokemon, dod_tmp_add:int = 0) -> bool:
    if odds_produce(opponent.dod + opponent.tmp_dod + dod_tmp_add):
        print(f"{opponent.name} 闪避了这次攻击！")
        # 电被动判断
        if "电被动" in opponent.buff:
            buff_add(opponent, "额外行动", 1)
        return False
    else:
        return True

# 空白技能
def none(self:Pokemon, opponent:Pokemon) -> None:
    print(f"{self.name} 发呆了一个回合")

def Thunderbolt(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self,  "十万伏特")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.THUNDERBOLT_DAMAGE_RATIO, "electric")
        take_damage(self, opponent, real_damage)
        if odds_produce(constants.THUNDERBOLT_PARALYSIS_CHANCE):
            # 麻痹判定
            buff_add(opponent, "麻痹", 2)

def Quick_Attack(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "电光一闪")
    real_damage:float = skill_damage_calculate(self, opponent, constants.QUICK_ATTACK_DAMAGE_RATIO, "none")
    if odds_produce(constants.QUICK_ATTACK_DOUBLE_HIT_CHANCE):
        print("你触发了双重攻击！")
        for x in range(2):
            if dod_judgement(opponent):
                take_damage(self, opponent, real_damage)
    else:
        if dod_judgement(opponent):
            take_damage(self, opponent, real_damage)

def Seed_Bomb(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "种子炸弹")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.SEED_BOMB_DAMAGE_RATIO, "grass")
        take_damage(self, opponent, real_damage)
        if odds_produce(constants.SEED_BOMB_POISON_CHANCE):
            buff_add(opponent, "中毒", 3)

def Parastitic_Seeds(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "寄生种子")
    buff_add(opponent, "种子寄生", 3)

def Aqua_jet(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "水枪")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.AQUA_JET_DAMAGE_RATIO, "water")
        take_damage(self, opponent, real_damage)

def Shield(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "护盾")
    buff_add(self, "护盾", 1)

def Ember(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "火花")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.EMBER_DAMAGE_RATIO, "fire")
        take_damage(self, opponent, real_damage)  
        # 烧伤判定
        if odds_produce(constants.EMBER_BURN_CHANCE):
            buff_add(opponent, "烧伤", 2)

def Flame_Charge(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "蓄能爆炎")
    if "爆炎蓄能完毕" in self.buff:
        self.buff.pop("爆炎蓄能完毕")
        if dod_judgement(opponent, constants.FLAME_CHARGE_TMP_DOD):
            real_damage:float = skill_damage_calculate(self, opponent, constants.FLAME_CHARGE_DAMAGE_RATIO, "fire")
            take_damage(self, opponent, real_damage)
            if odds_produce(constants.FLAME_CHARGE_BURN_CHANCE):
                        buff_add(opponent, "烧伤", 2)
    else:
        print("爆炎蓄能中")
        self.buff["爆炎蓄能完毕"] = -1

def Hyper_Beam(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self,  "破坏光线")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.HYPER_BEAM_DAMAGE_RATIO, "none")
        take_damage(self, opponent, real_damage)
        buff_add(self, "无法行动", 1)

def Hydro_Pump(self:Pokemon, opponent:Pokemon) -> None:
    skill_use_print(self, "水炮")
    if dod_judgement(opponent):
        real_damage:float = skill_damage_calculate(self, opponent, constants.HYDRO_PUMP_DAMAGE_RATIO, "water")
        take_damage(self, opponent, real_damage)
        if odds_produce(constants.HYDRO_PUMP_DEBUFF_CHANCE):
            buff_add(opponent, "闪避减少 15%", 1)


SKILL_FUNCTION_DICT = {
    "十万伏特":Thunderbolt, "电光一闪":Quick_Attack,
    "种子炸弹":Seed_Bomb, "寄生种子":Parastitic_Seeds,
    "水枪":Aqua_jet, "护盾":Shield,
    "火花":Ember, "蓄能爆炎":Flame_Charge,
    "破坏光线":Hyper_Beam, "水炮":Hydro_Pump,
    "发呆":none}