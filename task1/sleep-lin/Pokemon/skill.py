'''这个文档是技能系统'''
import random

class Skill:
    def __init__(self, name: str, power: float, target: str, effect: str = None, effect_chance: float = 0.0, duration: int = None, effect_description: str = ""):
        self.name = name
        self.power = power  # 技能伤害倍率
        self.target = target  # 施加对象，可以是对手或者自己
        self.effect = effect  # 技能的特殊效果，如麻痹、烧伤等
        self.effect_chance = effect_chance  # 特殊效果触发几率
        self.duration = duration  # 持续回合数
        self.effect_description = effect_description  # 技能描述

    def deal_damage(self, attacker, defender):
        print(f"{attacker.name} 使用了 {self.name}!")
        if self.power is not None:
            damage = int(attacker.attack * self.power)  # 基于攻击力和倍率计算伤害
            defender.take_damage(damage,attacker.attribute)

    def apply_effect(self, attacker, target):
        # 检查是否触发特殊效果            
         if self.effect and random.random() < self.effect_chance / 100:
            if self.target == '自己':
                print(f"{attacker.name} 获得了 {self.effect} 效果！")
                attacker.apply_status_effect(self.effect, self.duration)
            elif self.target == '对手':
                print(f"{target.name} 获得了 {self.effect} 效果！")
                target.apply_status_effect(self.effect, self.duration)
    
    def change(self,tianfa,attribute):
        if attribute == '无':
            return 
        if self.name == '普通公鸡' and random.random()<0.1:
            tianfa.attribute = attribute
            print(f'普通公鸡的特殊效果触发了，天罚的属性变为了{attribute}!')
        elif self.name == '拟态':
            tianfa.attribute = attribute
            print(f'天罚的属性变为了{attribute}!')
        else :
            return
            

# 定义技能
thunderbolt = Skill('十万伏特', 1.4, '对手', '麻痹', 10, 1, "对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人麻痹")
quick_attack = Skill('电光一闪', 1.0, '对手', '双重打击', 10, 1, "对敌人造成 1.0 倍攻击力的快速攻击（快速攻击有几率触发第二次攻击），10% 概率触发第二次")
seed_bomb = Skill('种子炸弹', 1.0, '对手', '中毒', 100, 100, "妙蛙种子发射一颗种子，爆炸后对敌方造成草属性伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失10%生命值")
parasitic_seeds = Skill('寄生种子', 1.0, '对手', '寄生', 100, 3, "妙蛙种子向对手播种，每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合缓慢吸取对手的生命值")
water_gun = Skill('水枪', 1.4, '对手', None, None,None,"杰尼龟喷射出一股强力的水流，对敌方造成 140% 水属性伤害")
shield = Skill('护盾', None, '自己','减伤',100,1, "杰尼龟使用水流形成保护盾，减少下一回合受到的伤害50%")
ember = Skill('火花', 1.0, '对手', '烧伤', 10, 2, "小火龙发射出一团小火焰，对敌人造成 100% 火属性伤害，并有10%的几率使目标陷入“烧伤”状态（每回合受到10额外伤害， 持续2回合）")
flame_charge = Skill('蓄能爆炎',None, '对手', '蓄能炎爆', 100, 2, "小火龙召唤出强大的火焰，对敌人造成 300% 火属性伤害，并有80%的几率使敌人陷入“烧伤”状态，这个技能需要1个回合的蓄力")
ordinary_rooster = Skill('普通公鸡', 1.2, '自己', None, 10, 1, "召唤一只公鸡，对敌人造成基于攻击力 120% 的伤害，有 10% 的概率变为克制对方的属性")
mimicry = Skill('拟态', None, '自己', None, 100, 100, "变为克制对方属性的属性")