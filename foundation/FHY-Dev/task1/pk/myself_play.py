from __future__ import annotations
import random
import sys

import myself_pokemon
from myself_pokemon import Pokemon
from myself_skills import FlameCharge


class ExitGame(Exception):
    pass


def exitgame(s: str) -> None:
    '''主动退出游戏'''
    if s.strip().lower() in ['exit', 'quit', '退出']:
        raise ExitGame
   

class Play:
    def __init__(self, all_pokemon: list) -> None:
        self.all_pokemon: list['Pokemon'] = all_pokemon
        self.not_choose_pokemon: list['Pokemon'] = all_pokemon.copy()
        self.you_pokemon: 'Pokemon' = None
        self.com_pokemon: 'Pokemon' = None
        self.you_pokemon_team: list['Pokemon'] = []
        self.com_pokemon_team:list['Pokemon'] = []
        for p in all_pokemon:
            p.play = self

    def is_you(self, pokemon: Pokemon) -> bool:
        '''判断该宝可梦是否属于你，属于你则输出 True'''
        if pokemon == self.you_pokemon:
                return True
        return False

    def print_pokeomon(self, lst: list) -> None:
        '''打印宝可梦列表'''
        No = 1
        for i in lst.copy():
            print(f'{No}. {i}', end = ' ')
            No += 1
        print()
    
    def you_choose_pokemon_team(self) -> None:
        '''我方选择两个宝可梦进队，
        赋值 self.youpokemon.team；
        并删除 self.not_chhoose_pokemon 中的部分元素
        '''
        self.print_pokeomon(self.all_pokemon)
        you_choose_num = input('请选择两只宝可梦组成你的队伍：')

        # 判断输入是否正确
        while(1):
            including = 1
            you_choose_num = you_choose_num.split()
            for i in you_choose_num:
                exitgame(i)
            try:
                you_choose_num = [int(j) for j in you_choose_num]
            except:
                you_choose_num = input('请确保您的输入为数字：')
                continue
            for i in you_choose_num:
                if i <= 0 or i > 5:
                    including = 0
                    break
            if not including:
                you_choose_num = input('请正确输入宝可梦的序号：')
                continue
            if len(you_choose_num) == 2:
                break
            else:
                you_choose_num = input(('请确保您的输入包含两个数字：'))
        
        you_choice = [self.not_choose_pokemon[i-1]
                      for i in you_choose_num]
        self.you_pokemon_team.extend(you_choice)
        for obj in you_choice:
            self.not_choose_pokemon.remove(obj)
        print(f'你选择了', end = ' ')
        for i in self.you_pokemon_team.copy():
            print(f'{i}', end = ' ')
        print('')
        
    def com_choose_pokemon_team(self) -> None:
        '''电脑选择两个宝可梦进队，
        赋值 self.com_pokemon_team；
        并删除 self.not_choose_pokemon 中的部分元素
        '''
        for i in range(1,3):
            n = random.choice(range(0, len(self.not_choose_pokemon)))
            self.com_pokemon_team.append(self.not_choose_pokemon.pop(n))
        print(f'电脑选择了',end = ' ')
        for i in self.com_pokemon_team.copy():
            print(f'{i}',end = ' ')
        print()

    def you_choose_pokemon(self) -> None:
        '''我方选择宝可梦，赋值 self.you_pokemon；
        并删除 self.you_pokemon_team 中的部分元素
        '''
        print('你的队伍有')
        self.print_pokeomon(self.you_pokemon_team)
        n = input('选择你的宝可梦：')

        # 判断输入是否正确
        while(1):
            exitgame(n)
            try:
                n= int(n)
            except:
                n = input('请确保你的输入为一个数字：')
                continue
            if n <= 0 or n > 2:
                n = input('请正确输入宝可梦序号：')
                continue
            else:
                break

        self.you_pokemon = self.you_pokemon_team.pop(n-1)
        print(f'就是你了，{self.you_pokemon}！')


    def com_choose_pokemon(self) -> None:
        '''电脑选择宝可梦，赋值 self.com_pokemon'''
        n = len(self.com_pokemon_team)
        No = random.choice(range(0, n))
        self.com_pokemon = self.com_pokemon_team.pop(No)
        print(f'电脑派出了{self.com_pokemon}')

    def print_skills(self, skills: list) -> None:
        '''打印技能组'''
        No = 1
        for i in skills.copy():
            print(f'{No}. {i}',end = ' ')
            No += 1
            print()

    def you_choose_skill(self) -> None:
        '''我方选择技能，我方 use_skill'''
        # 如果存在 after_charge 则无法正常选择技能
        during_skill_status = [
            i for i in self.you_pokemon.statuses.keys()
            if i.is_during_skill and i.name == 'after charge'
            ]
        if during_skill_status:
            self.you_pokemon.use_skill(skill = FlameCharge(),
                                       opponent = self.com_pokemon)
            return None

        print(f'你的 {self.you_pokemon} 的技能：')
        self.print_skills(self.you_pokemon.skills)
        index = input('选择一个技能进行攻击：')
        # 判断输入是否正确
        exitgame(index)
        while(1):
            try:
                index = int(index)
            except:
                index = input('请确保您的输入为数字：')
                continue
            if index <= 0 or index > len(self.you_pokemon.skills):
                index = input('请输入正确的技能编号：')
                continue
            else:
                break

        index_skill = self.you_pokemon.skills[index-1]
        self.you_pokemon.use_skill(skill = index_skill,
                                   opponent = self.com_pokemon)

    def com_choose_skill(self) -> None:
        '''对方选择技能，敌方 use_skill'''
        # 如果存在 after_charge 则无法正常选择技能
        during_skill_status = [
            i for i in self.com_pokemon.statuses.keys()
            if i.is_during_skill and i.name == 'after charge'
            ]
        if during_skill_status:
            self.com_pokemon.use_skill(skill = FlameCharge(),
                                       opponent = self.you_pokemon)
            return None

        index = random.choice([0, 1])
        index_skill = self.com_pokemon.skills[index]
        self.com_pokemon.use_skill(skill = index_skill,
                                   opponent = self.you_pokemon)
    
    def battle_prepare(self) -> None:
        '''战前准备'''
        self.you_choose_pokemon_team()
        self.com_choose_pokemon_team()
        print('\n')
        self.you_choose_pokemon()
        self.com_choose_pokemon()

    def print_pokemon_status(self) -> None:
        '''打印状态'''
        print(f'{self.you_pokemon}剩余{self.you_pokemon.hp}点血量')
        for i in self.you_pokemon.statuses.keys():
            if not i.is_during_skill and not i.is_after_skill:
                print(f'处于{i}状态中')

        print(f'{self.com_pokemon}剩余{self.com_pokemon.hp}点血量')
        for i in self.com_pokemon.statuses.keys():
            if not i.is_during_skill and not i.is_after_skill:
                print(f'处于{i}状态中')

    def before_skill_time(self) -> None:
        '''技能释放前，结算部分异常状态'''
        self.you_pokemon.before_skill(self.com_pokemon)
        self.com_pokemon.before_skill(self.you_pokemon)
 
    def battle_round(self) -> None:
        '''回合开始'''
        print('--------')
        self.print_pokemon_status()
        self.before_skill_time()
        self.you_choose_skill()
        self.com_choose_skill()
        
    def is_alive(self) -> bool:
        '''判断双方宝可梦是否都存活，都存活输出 True'''
        if (not self.you_pokemon.hp <= 0
        and not self.com_pokemon.hp <= 0):
            return True
        return False

    def exchange_current_pokemon(self) -> None:
        '''下场濒死宝可梦，上场存活宝可梦
        重新赋值 you_pokemon 或 com_pokemon
        '''
        if self.you_pokemon.hp <= 0:
            print(f'你的{self.you_pokemon}陷入濒死状态...')
            print(f'你收回了{self.you_pokemon}。')
            self.you_choose_pokemon()
        if self.com_pokemon.hp <= 0:
            print(f'电脑的{self.com_pokemon}陷入濒死状态...')
            print(f'电脑收回了{self.com_pokemon}。')
            self.com_choose_pokemon()

    def after_round_time(self) -> None:
        '''回合结束，结算部分异常状态'''
        self.you_pokemon.after_round(self.com_pokemon)
        self.com_pokemon.after_round(self.you_pokemon)

    def you_lose(self) -> bool:
        if not self.you_pokemon_team and self.you_pokemon.hp <= 0:
            return True
        return False

    def com_lose(self) -> bool:
        if not self.com_pokemon_team and self.com_pokemon.hp <= 0:
            return True
        return False

    def game_over(self) -> bool:
        '''判断是否对局是否结束，结束则输出 True'''
        if self.you_lose() or self.com_lose():
            return True
        return False

    def win(self) -> None:
        '''判断赢家'''
        if self.you_lose() and self.com_lose():
            print('你又赢又输')
        else:
            if self.you_lose():
                print('你输了...')    
            if self.com_lose():
                print('你赢了！！！')
    
    def run(self) -> None:
        '''游戏主循环'''
        try:
            self.battle_prepare()
            while not self.game_over():
                self.battle_round()
                if self.game_over():
                    break
                elif not self.is_alive():
                    self.exchange_current_pokemon()
                self.after_round_time()
                if self.game_over():
                    break
                elif not self.is_alive():
                    self.exchange_current_pokemon()
            self.win()
        except  ExitGame:
            print('玩家退出了游戏')


if __name__ == '__main__':
    pokemons = [myself_pokemon.Bulbasaur(),
                myself_pokemon.PikaChu(),
                myself_pokemon.Squirtle(),
                myself_pokemon.Charmander(),
                myself_pokemon.Smeargle()]
    game = Play(pokemons)
    game.run()