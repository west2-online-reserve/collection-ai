# 定义pokemon基类

import random
import skills
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from play import *

class Pokemon:

    # 属性宝可梦重写
    ATTR='none'
    DISADV=[]
    ADV=[]

    # 具体宝可梦重写
    MAX_HP=0
    DODGE=0

    # 技能表 具体宝可梦重写
    skillList=[]

    def __init__(self,attack,defence):
        self.hp=self.MAX_HP
        self.attack=attack
        self.defence=defence
        self.dodge=self.DODGE

        self.attackGain=1.0 # 伤害增益率 
        self.attacked=1.0 # 减少受到伤害的增益
        self.attackedTime=0 # 减伤增益时间，0时增益回初始1.0

        self.prepared=0 # 蓄力值

    def negative(self,attackValue:float,play:'Play',gamer:str):
        "属性被动"
        raise NotImplemented

    def reduceHp(self,attackValue):
        "扣血"
        if attackValue>0:
            self.hp-=attackValue
        if self.hp<0:
            self.hp=0

    def restoreHp(self,value):
        self.hp+=value
        if self.hp>self.MAX_HP:
            self.hp=self.MAX_HP

    def isFaint(self) -> bool:
        if self.hp<=0:
            return True
        else:
            return False
        
    def getDodge(self):
        return self.dodge

    def getAttack(self):
        return self.attack

    def isMiss(self,dodgeUp:float) -> bool:
        "自己是否闪避"
        return random.random()<self.getDodge()+dodgeUp

    def isSkillNumValid(self,num:str,user:Pokemon) -> bool:
        "判断选择的技能编号是否合法"
        try:
            numInt=int(num)
        except BaseException as e:
            return False
        else:
            if not 1<=numInt<=len(user.skillList):
               return False
            else:
                return True 
        
    def chooseSkill(self,fighter:Pokemon) -> type[skills.Skill]:
        "选择技能"
        print('技能如下：')
        cnt=1
        for skill in fighter.skillList:
            print(f'{cnt}.{skill.name}',end=' ')
            cnt+=1
        print()

        choice=input('请选择要使用的技能的编号:')
        if not self.isSkillNumValid(choice,fighter):
            choice=input('请输入正确的编号:')

        return fighter.skillList[int(choice)-1]
    
    def chooseSkillOpponent(self,fighter:Pokemon):
        "对手随机选择技能"
        num=random.sample(range(0,len(fighter.skillList)),1)[0]
        return fighter.skillList[num]

    def useSkill(self,skill:type[skills.Skill],play:'Play',user:str) -> tuple:
        "使用技能，如有伤害算上攻击修正率和宝可梦增效"
        print(f'\n{play.TEXT[user]}的 {self.name} 使用了{skill.name}!')
        result=list(skill.cast(self))
        result[0]=result[0]*play.attackRate[user]*self.attackGain

        return result

    def beAttacked(self,result:tuple,play:'Play',attacked:str) -> bool:
        "被攻击并扣血,算上防御和减伤，先算减伤（百分比），再算防御（减去）,成功闪避返回False"
        once=False

        if len(result)>=3 and result[2]:
            result[2]=False
            once=self.beAttacked(result,play,attacked)

        if len(result)==4:
            dodgeUp=result[3]
        else:
            dodgeUp=0
                
        if not self.isMiss(dodgeUp):
            if result[0]!=0:
                trueAttackValue=result[0]*self.attacked

                # 受击被动
                if self.ATTR=='water':
                    trueAttackValue=self.negative(trueAttackValue,play,attacked)

                trueAttackValue-=self.defence

                if trueAttackValue<=0:
                    trueAttackValue=0

                self.reduceHp(round(trueAttackValue))
                print(f'{play.TEXT[attacked]}的 {self.name} 受到了{round(trueAttackValue)}点伤害!',end=' ')
                print(f'{play.TEXT[attacked]}的 {self.name} HP: {self.hp}/{self.MAX_HP}')

            # 施加状态
            if result[1]:
                play.effect[attacked].append(result[1])

                print(f'{play.TEXT[attacked]}的 {self.name} {result[1].name}了！')

            if result[0]:
                return True
            else:
                return False
        else:
            print(f'但是{play.TEXT[attacked]}的 {self.name} 躲开了!')
            return False or once
        
        
# 定义含有属性的宝可梦

class GlassPokemon(Pokemon):
    ATTR='glass'
    ATTR_CN='草属性'

    DISADV=['fire']
    ADV=['water']

    # 属性被动
    
    def negative(self,attackValue:float,play:'Play',gamer:str):
        "每回合回复 10% 最大 HP 值"
        self.restoreHp(round(0.1*self.MAX_HP))
        print(f'\n{play.TEXT[gamer]}的 {self.name} 草系被动触发! 回复了 {round(0.1*self.MAX_HP)} 点血量! HP: {self.hp}/{self.MAX_HP}')
        

class FirePokemon(Pokemon):
    ATTR='fire'
    ATTR_CN='火属性'

    DISADV=['water']
    ADV=['glass']

    # 属性被动

    def negative(self,attackValue:float,play:'Play',gamer:str):
        "每次造成伤害，叠加 10% 攻击力，最多 4 层"
        if self.attackGain<1.4:
            self.attackGain+=0.1

            print(f'{play.TEXT[gamer]}的 {self.name} 火属性被动触发!伤害提升了!')
            
class WaterPokemon(Pokemon):
    ATTR='water'
    ATTR_CN='水属性'

    DISADV=['glass','elec']
    ADV=['fire']

    # 属性被动
    
    def negative(self,attackValue:float,play:'Play',gamer:str) -> float:
        "受到伤害时，有 50% 的几率减免 30% 的伤害"
        "受伤时调用"
        
        luck=random.randint(0,1)

        if luck:
            print(f'{play.TEXT[gamer]}的 {self.name} 触发了被动技能!展开了一次水盾!')
            return attackValue*(1-0.3)
        else:
            return attackValue


class ElecPokemon(Pokemon):
    ATTR='elec'
    ATTR_CN='电属性'

    DISADV=[]
    ADV=['water']

    # 属性被动

    def negative(self,attackValue:float,play:'Play',gamer:str):
        "当成功躲闪时，可以立即使用一次技能"
        print(f'{play.TEXT[gamer]}的 {self.name} 触发电属性被动！{self.name} 反击！')


        

            
        
        