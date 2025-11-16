# 定义游戏规则

from pokemon import Pokemon
from detailed_pokemon import POKEMON_LIST
import detailed_pokemon 
import effects

import random
import sys
import time

class Play:
    def __init__(self):
        self.CYCLE_LIST=['player','opponent']
        self.TEXT={'player':'你','opponent':'对手'}
        self.OTHER={'player':'opponent','opponent':'player'} 
        self.actOrder=[] # 行动顺序

        self.teams={'player':[],'opponent':[]} # 宝可梦队伍
        
        self.TEAM_SIZE=3 # 队伍大小

        self.currentPokemonDict={'player':None,'opponent':None} # 正在战斗的宝可梦

        self.turn=1 # 回合数

        self.attackRate={"player":1,'opponent':1} # 攻击修正率
        self.firePokemonStrength={'player':1,'opponent':1} # 最大为1.4,第一个玩家，第二个对手

        self.isDisabled={'player':False,'opponent':False} # 是否无法行动
        self.isMore={'player':False,'opponent':False} # 是否是额外行动
        self.effect={'player':[],'opponent':[]}

    def drawLine(self,length:int=30):
        "画个线"
        for i in range(0,length):
            print('-',end='')
        print()

    def run(self):

        # 选择宝可梦
        self.choosePokemon()
        
        # 战斗
        self.battle()

        sys.exit()
        
    def isPokemonNumValid(self,numList:list) -> bool:
        "判断选择的宝可梦编号列表是否合法"
        try:
            for num in numList:
                numInt=int(num)
                if not 1<=numInt<=len(POKEMON_LIST):
                    return False
        except BaseException as e:
            return False
                
        return True

    def choosePokemon(self) -> list:
        "选择宝可梦队伍函数,各方选pokemonNum个，对手随机选择"

        # 对手选择
        self.teams['opponent']=[POKEMON_LIST[memberNum]() for memberNum in random.sample(range(0,len(POKEMON_LIST)),self.TEAM_SIZE)]

        print(f'请选择 {self.TEAM_SIZE} 个宝可梦组成你的队伍(编号用空格分开):')
        print('\t',end='')
        cnt=1
        for i in POKEMON_LIST:
            print(f'{cnt}.{i.name}({i.ATTR_CN})',end=' ')
            cnt+=1
        print()

        # 玩家选择
        members = [x for x in input('请输入宝可梦编号:').strip().split() if x]
        jdg=True
        while jdg:
            jdg=False
            while not self.isPokemonNumValid(members):
                members = [x for x in input('请输入正确的宝可梦编号:').strip().split() if x]

            members=list(map(int,members))
            if len(members)>self.TEAM_SIZE:
                members = [x for x in input('宝可梦过多:').strip().split() if x]
                jdg=True
            elif len(members)<self.TEAM_SIZE:
                members = [x for x in input('宝可梦过少:').strip().split() if x]
                jdg=True

        self.teams['player']=[POKEMON_LIST[number-1]() for number in members]

        self.drawLine()
        print('你选择了如下宝可梦队伍：')
        print('\t',end='')
        cnt=1
        for member in self.teams['player']:
            print(f'{cnt}.{member.name}({member.ATTR_CN})',end=' ')
            cnt+=1
        print()

        time.sleep(1)

        print(f'你的对手也选择了{self.TEAM_SIZE}个宝可梦')
        time.sleep(1)
        
    def initActOrder(self):
        "初始每回合的行动队列，默认我方先手"
        self.actOrder.append('player')
        self.actOrder.append('opponent')

    def initAttackValue(self):
        "初始化攻击值栈"
        self.attackValue=0

    def isPokemonOrderValid(self,orderNum:str) -> bool:
        "判断指派战斗的宝可梦编号是否合法"
        try:
            num=int(orderNum)
        except BaseException as e:
            return False
        else:
            if num>len(self.teams['player']):
                return False
            else:
                return True
        
    def orderPokemon(self):
        "玩家指派出战宝可梦"
        self.drawLine()
        print('你的队伍:')
        print('\t',end='')

        cnt=1
        for member in self.teams['player']:
            print(f'{cnt}.{member.name}({member.ATTR_CN})',end=' ')
            cnt+=1
        print()
        
        num=input('请选择出战宝可梦编号:')
        while not self.isPokemonOrderValid(num):
            num=input('请输入正确的编号:')
        
        self.currentPokemonDict['player']=self.teams['player'][int(num)-1]
        print(f'你选择了{self.currentPokemonDict['player'].name}!')

    def opponentOrderPokemon(self):
        "对手指派出战宝可梦，默认按对手宝可梦队列顺序出战"
        self.currentPokemonDict['opponent']=self.teams['opponent'][0]
        print(f'对手派出了{self.currentPokemonDict['opponent'].name}!')

    def multifyAttackRate(self,target:str,rate:float):
        "乘算攻击修正率，填入修正率则原修正率=原修正率*rate"
        if target=='player':
            self.attackRate['player']*=rate
        elif target=='opponent':
            self.attackRate['opponent']*=rate
        else:
            raise

    def changeAttackRate(self):
        player:Pokemon=play.currentPokemonDict['player']
        opponent:Pokemon=play.currentPokemonDict['opponent']
        
        # player attack
        if opponent.ATTR in player.ADV:
            self.multifyAttackRate('player',2.0)
        elif opponent.ATTR in player.DISADV:
            self.multifyAttackRate('player',0.5)

        # opponent attack
        if player.ATTR in opponent.ADV:
            self.multifyAttackRate('opponent',2.0)
        elif player.ATTR in opponent.DISADV:
            self.multifyAttackRate('opponent',0.5)

    def act(self,chooseSkillDict:dict,gamer:str,battlePokemon:Pokemon,opponentPokemon:Pokemon) -> bool:
        "行动:选择技能，使用技能，可能使对手受到伤害,对手闪避返回False，否则True"

        if gamer=='player':
            if self.isMore[gamer]:
                print('电属性闪避成功,电光一闪，你可以再次行动！')
                print('================你的额外回合================')
                
            self.drawLine()
            print('对手的宝可梦:')
            print(f'\t{opponentPokemon.name} HP: {opponentPokemon.hp}/{opponentPokemon.MAX_HP}')
            print('你当前的宝可梦：')
            print(f'\t{battlePokemon.name} HP: {battlePokemon.hp}/{battlePokemon.MAX_HP}')

        skill=chooseSkillDict[gamer](battlePokemon)
        attackValue=battlePokemon.useSkill(skill,self,gamer)
        isAttacked=opponentPokemon.beAttacked(attackValue,self,self.OTHER[gamer])

        return isAttacked

    def battle(self):
        "一次回合制战斗循环，以其中一方全员倒下为止"
        
        # 双方选择出战宝可梦
        self.orderPokemon()
        time.sleep(1)
        self.opponentOrderPokemon()
        time.sleep(1)

        # change修正率
        self.changeAttackRate()

        ORDER_DICT={'player':self.orderPokemon,'opponent':self.opponentOrderPokemon}

        # 开始战斗循环
        while len(self.teams['player'])>0 and len(self.teams['opponent'])>0:
            # 初始化本回合的行动顺序
            self.initActOrder()

            # 回合中的两方行动
            for gamer in self.actOrder: # gamer is str
                battlePokemon:Pokemon=self.currentPokemonDict[gamer]
                opponentPokemon:Pokemon=self.currentPokemonDict[self.OTHER[gamer]]
                CHOOSE_SKILL_DICT={'player':battlePokemon.chooseSkill,'opponent':battlePokemon.chooseSkillOpponent}

                if gamer=='player':    
                    print('\n================你的回合================')

                # 增益消失
                if battlePokemon.attackedTime==1:
                    battlePokemon.attackedTime-=2
                elif battlePokemon.attackedTime==-1:
                    print(f"\n{play.TEXT[gamer]}的 {battlePokemon.name} 的防御技能失效了！")
                    battlePokemon.attacked=1.0
                    battlePokemon.attackedTime=0

                # 遍历状态列表
                if (not battlePokemon.isFaint()) and (not self.isMore[gamer]):
                    # 消除
                    i=0
                    while i<len(self.effect[gamer]):
                        if self.effect[gamer][i].time>0:
                            i+=1
                        else:
                            print(f'\n{play.TEXT[gamer]}的 {battlePokemon.name} {self.effect[gamer][i].name}状态解除了!')
                            if self.effect[gamer][i].name in effects.disAbleEffectList:
                                self.isDisabled[gamer]=False
                                battlePokemon.dodge=battlePokemon.DODGE
                            self.effect[gamer].pop(i)

                    # 发作
                    for effect in self.effect[gamer]:
                        effect.cast(battlePokemon,gamer,self)
                        effect.timeGo()
                        time.sleep(1)

                # 判断宝可梦是否昏迷
                if battlePokemon.isFaint():
                    # 排除出队伍
                    index=self.teams[gamer].index(battlePokemon)
                    self.teams[gamer].pop(index)

                    self.effect[gamer].clear()

                    print(f'\n{play.TEXT[gamer]}的 {battlePokemon.name} 倒下了！')

                    # 指派宝可梦
                    if len(self.teams[gamer])<=0:
                        break

                    ORDER_DICT[gamer]()
                    battlePokemon=self.currentPokemonDict[gamer]
                    # 更改修正率
                    self.changeAttackRate()
                
                # 战斗开始时机的被动技能
                if battlePokemon.ATTR=='glass':
                    battlePokemon.negative(0,self,gamer)

                # 行动:选择技能，使用技能，可能使对手受到伤害
                if not self.isDisabled[gamer]:
                    isOpponentAttacked=self.act(CHOOSE_SKILL_DICT,gamer,battlePokemon,opponentPokemon) # 受击被动内嵌在该函数的beAttacked
                    
                    if isOpponentAttacked: # 成功攻击时机被动技能
                        if battlePokemon.ATTR=='fire':
                            battlePokemon.negative(0,self,gamer)
                    else: # 对方躲闪被动技能
                        if opponentPokemon.ATTR=='elec' and (not self.isDisabled[play.OTHER[gamer]]): # 再次行动，但反击被躲开对方不能再反击
                            gamer=play.OTHER[gamer]
                            self.isMore[gamer]=True
                            battlePokemon,opponentPokemon=opponentPokemon,battlePokemon

                            battlePokemon.negative(0,self,gamer)

                            isOpponentAttacked=self.act(CHOOSE_SKILL_DICT,gamer,battlePokemon,opponentPokemon)
                            self.isMore[gamer]=False

                            if isOpponentAttacked: # 成功反击时机被动技能
                                if battlePokemon.ATTR=='fire':
                                    battlePokemon.negative(0,self,gamer)
                else:
                    print(f'{play.TEXT[gamer]}的 {battlePokemon.name} 麻痹了，不能行动!')

                time.sleep(1)
            
            self.actOrder.clear()

            if len(self.teams['player'])<=0 or len(self.teams['opponent'])<=0:
                break
            else:
                self.turn+=1

        if len(self.teams['player'])>0:
            print('你获得了胜利！')
        else:
            print('你被击败了！')


    def attacked(self,attackValue:float,target:Pokemon):
        "扣血函数，把最后计算完的真实伤害(取整)扣在目标上"
        target.reduceHp(attackValue)
 
    def getFirePokemonStrength(self,target:str) -> int:
        return self.firePokemonStrength[target]
    
    def addFirePokemonStrength(self,target:str):
        self.firePokemonStrength[target]+=0.1

if __name__=='__main__':
    play=Play()
    play.run()

