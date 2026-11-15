from __future__ import annotations
import copy
import random
import sys

import pokemon
from pokemon import Pokemon


def valid_choice(choice, range):
    # 判断输入的选择是否合法
    return choice.isdigit() and 1 <= int(choice) <= range


class Play:
    def __init__(self, all_pokemon) -> None:
        self.all_pokemon = all_pokemon
        self.player_team :list["Pokemon"]= []
        self.computer_team :list["Pokemon"]= []
        self.current_player_pokemon = None
        self.current_computer_pokemon = None
        self.turn = 0

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=3)->None:
        # 玩家选择队伍中的 Pokemon
        print(f"选择 {num} 只宝可梦作为你的队伍:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)#浅拷贝
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"输入数字选择你的第 {index + 1} 只宝可梦：")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("无效的选择，请选择有效的宝可梦")
        print("这是你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=3)->None:
        # 电脑选择对战的队伍
        print(f"你的对手正在选择 {num} 只宝可梦")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("这是对手的队伍：")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印 Pokemon 列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的 Pokemon
        print("你的队伍：")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("输入数字选择你要出战的宝可梦：")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"你选择了 {chosen_pokemon.name}！")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的 Pokemon
                else:
                    print(f"{chosen_pokemon.name} 已经倒下了！请选择其他宝可梦")
            else:
                print("无效的选择，请选择有效的宝可梦")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的 Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"对手选择了 {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon  # 返回选择的 Pokemon

    def game_finished(self)->None:
        # 游戏结束
        sys.exit()

    def check_game_status(self)->bool:
        self._handle_deaths()
        # 检查游戏状态，判断玩家或电脑是否失败
        if not self.player_team and not self.computer_team:
            print("双方宝可梦全部倒下，平局！")
            self.game_finished()
            return True
        if not self.computer_team:
            print("你赢了！")
            self.game_finished()
            return True
        if not self.player_team:
            print("你输了！")
            self.game_finished()
            return True
    
        return False

    def _handle_deaths(self)->None:
        self.player_team = [p for p in self.player_team if p.alive and p.hp > 0]
        self.computer_team = [c for c in self.computer_team if c.alive and c.hp > 0]
        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            print(f"{self.current_player_pokemon.name} 倒下了！")
            if self.player_team:
                self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            print(f"对手的 {self.current_computer_pokemon.name} 倒下了！")
            if self.computer_team:
                self.computer_choose_pokemon()

    def player_use_skills(self)->None:
        # 玩家选择技能
        if self.current_player_pokemon is None:
            return
        print("选择你的宝可梦要使用的技能")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        choice = input("输入数字选择你想使用的技能：")
        if valid_choice(choice, len(skills)):
            player_skill = self.current_player_pokemon.skills[int(choice) - 1]
            self.current_player_pokemon.use_skill(
                player_skill, self.current_computer_pokemon
            )
        else:
            print("无效的选择，请选择有效的技能")

    def computer_use_skills(self)->None:
        # 电脑随机选择技能
        if self.current_computer_pokemon is None:
            return
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def battle_round_begin(self)->None:
        # 回合开始
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()

    def battle_round(self)->None:
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        # 回合进行
        print(
            f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}"
        )
        self.player_use_skills()
        if not self.current_computer_pokemon.alive:
            print(f"回合结束！{self.current_computer_pokemon.name} 倒下了！")
            return
        self.computer_use_skills()
        if not self.current_player_pokemon.alive:
            print(f"回合结束！{self.current_player_pokemon.name} 倒下了！")

    def run(self)->None:
        
        import copy
        player_pool = [copy.deepcopy(p) for p in self.all_pokemon]
        computer_pool = [copy.deepcopy(p) for p in self.all_pokemon]
        self.player_choose_pokemon_team(player_pool, num=3)
        self.computer_choose_pokemon_team(computer_pool, num=3)

        print("选择你的第一只宝可梦来战斗")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.turn += 1
            print(f"\n===== 回合 {self.turn} =====")
            self.battle_round_begin()
            self.battle_round()
            self._handle_deaths()
            if self.check_game_status():
                break


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.Charmander()
    pokemon3 = pokemon.PikaChu()
    pokemon4 = pokemon.Squirtle()
    all_pokemon = [pokemon1,pokemon2,pokemon3,pokemon4]
    play = Play(all_pokemon)
    play.run()
    
