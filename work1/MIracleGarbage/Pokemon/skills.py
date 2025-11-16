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
        
        