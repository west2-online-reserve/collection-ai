# 定义状态

# 更新disable表

from pokemon import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from play import *

class Effect:
    def __init__(self,time):
        self.time=time # 持续回合数

    def timeGo(self):
        "时间流逝，回合数-1"
        self.time-=1

    def cast(self, target:Pokemon,targetName:str,play:'Play'):
        raise NotImplemented

class Paralysis(Effect):
    "不能行动一回合"
    name="麻痹"

    def __init__(self,time=1):
        super().__init__(time)

    def cast(self, target:Pokemon,targetName:str,play:'Play'):
        play.isDisabled[targetName]=True
        target.dodge=0

            
class Poisoned(Effect):
    "中毒状态，每回合损失 10% 生命值"

    name='中毒'

    def __init__(self, time=3):
        super().__init__(time)

    def cast(self, target:Pokemon,targetName:str,play:'Play'):
        print()
        print(f'{play.TEXT[targetName]}的 {target.name} {self.name}了,受到{round(target.MAX_HP*0.1)}点伤害！',end=' ')
        target.reduceHp(round(target.MAX_HP*0.1))
        print(f'{play.TEXT[targetName]}的 {target.name} HP: {target.hp}/{target.MAX_HP}')
        
class Parasitized(Effect):
    "每回合被吸取 10% 的最大生命值并恢复对手，效果持续 3 回合"

    name="被寄生"

    def __init__(self, time=3):
        super().__init__(time)

    def cast(self, target:Pokemon,targetName:str,play:'Play'):
        print()
        print(f'{play.TEXT[targetName]}的 {target.name} {self.name}了,受到{round(target.MAX_HP*0.1)}点伤害！',end=' ')
        target.reduceHp(round(target.MAX_HP*0.1))
        print(f'{play.TEXT[targetName]}的 {target.name} HP: {target.hp}/{target.MAX_HP}')
        
        print(f'{play.TEXT[play.OTHER[targetName]]}的 {play.currentPokemonDict[play.OTHER[targetName]].name} 恢复{round(target.MAX_HP*0.1)}点HP！',end=' ')
        play.currentPokemonDict[play.OTHER[targetName]].restoreHp(round(target.MAX_HP*0.1))
        print(f'{play.TEXT[play.OTHER[targetName]]}的 {play.currentPokemonDict[play.OTHER[targetName]].name} HP: {play.currentPokemonDict[play.OTHER[targetName]].hp}/{play.currentPokemonDict[play.OTHER[targetName]].MAX_HP}')

class Fired(Effect):
    "「烧伤」状态（每回合受到 10 额外伤害，持续 2 回合）"
    name="烧伤"

    def __init__(self, time=2):
        super().__init__(time)

    def cast(self, target:Pokemon,targetName:str,play:'Play'):
        print()
        print(f'{play.TEXT[targetName]}的 {target.name} {self.name}了,受到{10}点伤害！',end=' ')
        target.reduceHp(10)
        print(f'{play.TEXT[targetName]}的 {target.name} HP: {target.hp}/{target.MAX_HP}')

# 会使宝可梦不能行动的状态
disAbleEffectList=[Paralysis.name]