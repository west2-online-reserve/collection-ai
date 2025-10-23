from __future__ import annotations
import copy
import random
import sys

import pokemon


def valid_choice(choice, range):
    # 判断输入的选择是否合法
    return choice.isdigit() and 1 <= int(choice) <= range


class Play:
    def __init__(self, all_pokemon) -> None:
        self.all_pokemon = all_pokemon
        self.player_team = []
        self.computer_team = []
        self.current_player_pokemon = None
        self.current_computer_pokemon = None
        self.turn = 0

    def choose_multiple_pokemon(self,pokemon_to_choose,num = 3):
        i = 1
        while i <= num:
            self.player_choose_pokemon_team(pokemon_to_choose,i)
            self.computer_choose_pokemon_team(pokemon_to_choose,i) 
            i += 1

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=1):
        # 玩家选择队伍中的Pokemon
        print(f"为你的队伍选择 {num} 只宝可梦:")
        pokemon_to_choose = copy.deepcopy(pokemon_to_choose)
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            print()
            choice = input(f"通过数字选择你的第 {index} 只宝可梦: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("无效选择，请选择一个有效的宝可梦")
        print("这是你的宝可梦队伍:")
        for i in self.player_team:
            i.operator = 'player'
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=1):
        # 电脑选择对战的队伍
        print(f"你的对手正在选择 {num} 只宝可梦")
        self.computer_team.extend(random.sample(pokemon_to_choose, 1))
        for i in self.computer_team:
            i.operator = 'computer'
        print("这是你对手的队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的Pokemon
        print("你的队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("通过数字选择你要出战的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"你选择了 {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    print(f"{chosen_pokemon.name} 已经战败! 请选择另一只宝可梦!")
            else:
                print("无效选择，请选择一个有效的宝可梦")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"你的对手选择了 {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon  # 返回选择的Pokemon

    def game_finished(self):
        # 游戏结束
        sys.exit()

    def check_game_status(self):
        
        # 检查游戏状态，判断玩家或电脑是否失败
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("你和对手的所有宝可梦都已战败。游戏平局。")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("电脑的所有宝可梦都已战败。你赢了!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("你的所有宝可梦都已战败。你输了!")
            self.game_finished()

        if not self.current_player_pokemon.alive:
            self.player_team.remove(self.current_player_pokemon)
            self.player_choose_pokemon()
        if not self.current_computer_pokemon.alive:
            self.computer_team.remove(self.current_computer_pokemon)
            self.computer_choose_pokemon()

            
    def player_use_skills(self):
        # 延迟技能使用
        if self.current_player_pokemon.delay_skill != None:
            player_skill = self.current_player_pokemon.delay_skill
            
        # 玩家选择技能   
        else:
            print("选择你的宝可梦要使用的技能")
            skills = self.current_player_pokemon.skills
            for i, skill in enumerate(skills, 1):
                print(f"{i}: {skill}")
            
            while True:
                choice = input("通过数字选择你想要使用的技能:")
                if valid_choice(choice, len(skills)):
                    player_skill = self.current_player_pokemon.skills[int(choice) - 1]
                    break
                else:
                    print("无效选择，请选择一个有效的技能")

        self.current_player_pokemon.use_skill(
            player_skill, self.current_computer_pokemon, self
        )
            
    def computer_use_skills(self):
        # 电脑随机选择技能
        if self.current_computer_pokemon.delay_skill != None:
            computer_skill = self.current_computer_pokemon.delay_skill
        else:
            computer_skill = random.choice(self.current_computer_pokemon.skills)

        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon, self
        )

    def take_effect(self):
        print('玩家状态')
        self.current_player_pokemon.apply_status_effect()
        print('------')
        print('电脑状态')
        self.current_computer_pokemon.apply_status_effect()
        print('------')

    def battle_round_begin(self):
        # 回合开始
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    def battle_round(self):
        # 回合进行
        print(
            f"\n-----{self.current_player_pokemon.name} 对战 {self.current_computer_pokemon.name}-----"
        )
        self.player_use_skills()
        self.take_effect()
        self.computer_use_skills()
        self.take_effect()
        self.check_game_status()

    def run(self):
        # 游戏主循环
        #self.player_choose_pokemon_team(self.all_pokemon)  
        #self.computer_choose_pokemon_team(self.all_pokemon)
        self.choose_multiple_pokemon(self.all_pokemon)

        print("选择你第一只要出战的宝可梦")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.PikaChu()
    pokemon3 = pokemon.Squirtle()
    pokemon4 = pokemon.Charmander()
    all_pokemon = [pokemon1,pokemon2,pokemon3,pokemon4]
    play = Play(all_pokemon)
    play.run()