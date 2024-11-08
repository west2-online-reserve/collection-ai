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

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=2):
        # 玩家选择队伍中的Pokemon
        print(f"为你的队伍选择 {num} 个宝可梦:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"按编号选择你下一个的宝可梦:  ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("无效选择,请选择一个有效的宝可梦")
        print("这是你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)
        print("\n")

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=2):
        # 电脑选择对战的队伍
        print(f"你的对手正在选择 {num} 号宝可梦 ")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("这是你对手的宝可梦队伍:")
        self.print_pokemon_list(self.computer_team)
        print("\n")

    def print_pokemon_list(self, pokemon_list):
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的Pokemon
        print("\n你的队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input(f"\n选择你将要上场的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"你选择了 {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    print(f"{chosen_pokemon.name} 已经进入睡眠状态,请选择其他宝可梦!")
            else:
                print("无效选择,请选择有效的宝可梦")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"\n你的对手选择了 {chosen_pokemon}")
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
            print("你和对手的所有宝可梦都昏迷了,游戏结束")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("你赢了!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("你输了!")
            self.game_finished()

        if not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        print(f"\n----现在是你的回合----")
        # 玩家选择技能
        print("选择你的宝可梦要使用的技能")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        choice = input("通过数字选择你的宝可梦将要使用的技能:")
        print("\n")
        if valid_choice(choice, len(skills)):
            player_skill = self.current_player_pokemon.skills[int(choice) - 1]
            self.current_player_pokemon.use_skill(
                player_skill, self.current_computer_pokemon
            )
        else:
            print("无效选择,请选择有效技能")
        print(f"--------------------")

    def computer_use_skills(self):
        # 电脑随机选择技能
        print(f"\n----现在是电脑的回合----")
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )
        print(f"---------------------")

    def battle_round_begin(self):
        # 回合开始
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    def battle_round(self):
        # 回合进行
        print(
            f"\n-----------------------\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}\n-----------------------\n"
        )
        self.player_use_skills()
        self.computer_use_skills()
        self.check_game_status()

    def run(self):
        # 游戏主循环
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        print("选择你第一个上场的宝可梦")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            print("\n")
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.Charmander()
    all_pokemon = [pokemon1, pokemon2]
    play = Play(all_pokemon)
    play.run()