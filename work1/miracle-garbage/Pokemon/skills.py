# 定义技能
from typing import TYPE_CHECKING

import effects

import random

if TYPE_CHECKING:
    from pokemon import Pokemon

class Skill:
    name='none'

    def __init__(self):
        pass

    @classmethod
    def cast(self,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        raise NotImplemented


class Thunderbolt(Skill):
    "对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人 麻痹"
    name='十万伏特'

    @classmethod
    def cast(self,player:'Pokemon') -> tuple:
        if random.random()<0.1:
            effect=effects.Paralysis()
        else:
            effect=None

        return (player.getAttack()*1.4,effect)
        
class QuickAttack(Skill):
    "对敌人造成 1.0 倍攻击力的快速攻击（快速攻击有几率触发第二次攻击），10% 概率触发第二次,被打到一次无法触发反击"
    name="电光一闪"

    @classmethod
    def cast(self,player:'Pokemon') -> tuple:
        effect=None

        if random.random()<0.1:
            twiceAttack=True
        else:
            twiceAttack=False

        return (player.getAttack()*1.0,effect,twiceAttack)

class SeedBomb(Skill):
    """
        妙蛙种子发射一颗种子，爆炸后对敌方造成草属性伤害。
        若击中目标，目标有 15% 几率陷入「中毒」状态，每回合损失 10% 生命值
    """
    name='种子炸弹'

    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        if random.random()<0.15:
            effect=effects.Poisoned()
        else:
            effect=None

        return (player.getAttack(),effect)
        

class ParasiticSeeds(Skill):
    "妙蛙种子向对手播种，每回合吸取对手 10% 的最大生命值并恢复自己，效果持续 3 回合"
    name='寄生种子'

    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        if random.random()<1:
            effect=effects.Parasitized()
        else:
            effect=None

        return (0,effect)
        
class AquaJet(Skill):
    "杰尼龟喷射出一股强力的水流，对敌方造成 140% 水属性伤害"
    name="水枪"
    
    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        effect=None

        return (player.getAttack()*1.4,effect)

class Shield(Skill):
    "杰尼龟使用水流形成保护盾，减少下一回合受到的伤害 50%"
    name="水盾"

    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        effect=None

        # 其他行为
        player.attacked-=0.5
        player.attackedTime=1

        return (0,effect)

class Ember(Skill):
    """
        小火龙发射出一团小火焰，对敌人造成 100% 火属性伤害，
        并有 10% 的几率使目标陷入「烧伤」状态（每回合受到 10 额外伤害，持续 2 回合）
    """
    name="火花"

    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        if random.random()<0.1:
            effect=effects.Fired()
        else:
            effect=None

        return (player.getAttack(),effect)
    
class FlameCharge(Skill):
    """
        小火龙召唤出强大的火焰，对敌人造成 300% 火属性伤害，
        并有 80% 的几率使敌人陷入「烧伤」状态，
        这个技能需要 1 个回合的蓄力，并且在面对该技能时敌法闪避率增加 20%
    """
    name="蓄能爆炎"

    @classmethod
    def cast(cls,player:'Pokemon') -> tuple:
        "返回技能的伤害与可能产生的状态"     
        if player.prepared==0:
            player.prepared=1
            effect=None
            return (0,effect)
        else:
            player.prepared=0
            if random.random()<0.8:
                effect=effects.Fired()
            else:
                effect=None
            return (player.getAttack()*3.0,effect,False,0.2)
        
