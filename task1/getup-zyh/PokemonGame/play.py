from __future__ import annotations
import copy
import random
import sys
from typing import List
import pokemon
from pokemon import Pokemon


def valid_choice(choice, range):
    # 判断输入的选择是否合法
    return choice.isdigit() and 1 <= int(choice) <= range


class Play:
    def __init__(self, all_pokemon) -> None:
        self.all_pokemon = all_pokemon
        self.player_team: List[Pokemon] = []
        self.computer_team: List[Pokemon] = []
        self.current_player_pokemon = None
        self.current_computer_pokemon = None
        self.turn = 0

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num:int):
        # 第二个参数是all_pokemon
        print(f"Choose {num} pokemon for your team:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"Select your pokemon {index} by number: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))#这段代码有点巧，.pop方法返回的是索引值，并且删除了原列表，一句两得
                index += 1
            else:
                print("Invalid choice, please select a valid Pokemon")
        print("Here is your pokemon team:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num:int):
        # 电脑选择对战的队伍
        print(f"Your opponent is choosing {num} pokemon")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("Here is your opponent's team:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的Pokemon
        print("Your Team:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("Select your pokemon to battle by number:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True: ##之所以任然保留这个逻辑，是因为增加代码健壮性
                    print(f"You choosed {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    print(f"{chosen_pokemon.name} has fainted! Choose another Pokemon!")
            else:
                print("Invalid choice, please select a valid Pokemon")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]#列表生成
        chosen_pokemon = random.choice(available_pokemon)
        print(f"Your opponent choosed {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon  # 返回选择的Pokemon

    def game_finished(self):
        # 游戏结束
        sys.exit()

    def check_game_status(self):
        # 检查游戏状态，判断玩家或电脑是否失败，先检查是否决出胜负，再检查当前pokemon状态，并是否移除
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("All your and opponent's Pokemon have fainted. The game is tied.")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("All computer's Pokemon have fainted. You win!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("All your Pokemon have fainted. You lose!")
            self.game_finished()

        if not self.current_player_pokemon.alive:
            print(f"{self.current_player_pokemon.name} has fainted!")
            self.player_team.remove(self.current_player_pokemon)  # 从玩家队伍移除死亡宝可梦，或许此处有点冗余
            if self.player_team:
                self.player_choose_pokemon()
            else:
                print("All your Pokémon have fainted. You lose!")
                self.game_finished()



        if not self.current_computer_pokemon.alive:
            print(f"{self.current_computer_pokemon.name} has fainted!")
            self.computer_team.remove(self.current_computer_pokemon)  # 从电脑队伍移除死亡宝可梦
            if self.computer_team:
                self.computer_choose_pokemon()
            else:
                print("All opponent's Pokémon have fainted. You win!")
                self.game_finished()

    def player_use_skills(self):
        # 玩家选择技能
        print("Choose the skill your pokemon to use")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        choice = input("Select the skill you want to use by number:")
        if valid_choice(choice, len(skills)):
            player_skill = self.current_player_pokemon.skills[int(choice) - 1]
            self.current_player_pokemon.use_skill(
                player_skill, self.current_computer_pokemon
            )
        else:
            print("Invalid choice, please select a valid Skill")

    def computer_use_skills(self):
        # 电脑随机选择技能
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def battle_round_begin(self):
        # 回合开始
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    def battle_round(self):
        print(f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}")

    # 玩家宝可梦使用技能攻击电脑宝可梦
        if self.current_player_pokemon.apply_status_effect():#apply_status_effect是根据paralyzed情况的布尔值
            self.player_use_skills()  # 玩家选择并使用技能
        else:
            print(f"{self.current_player_pokemon.name} is paralyzed and cannot move this turn!")

    # 检查电脑宝可梦是否还活着
        if not self.current_computer_pokemon.alive:
            print(f"{self.current_computer_pokemon.name} has fainted!")
            return  # 如果电脑宝可梦晕倒，回合结束

    # 电脑宝可梦使用技能攻击玩家宝可梦
        if self.current_computer_pokemon.apply_status_effect():
            self.computer_use_skills()  # 电脑选择并使用技能
        else:
            print(f"{self.current_computer_pokemon.name} is paralyzed and cannot move this turn!")

    # 检查玩家宝可梦是否还活着
        if not self.current_player_pokemon.alive:
            print(f"{self.current_player_pokemon.name} has fainted!")

    # 检查游戏状态，判断是否有一方胜利
        self.check_game_status()

    

    def run(self):
        # 游戏主循环
        batle_num = input("选择双方宝可梦数量:")
        if valid_choice(batle_num, len(all_pokemon)):
            num = int(batle_num)
        else:
            print("Invalid choice, please select a valid Pokemon")

        self.player_choose_pokemon_team(self.all_pokemon,num)
        self.computer_choose_pokemon_team(self.all_pokemon,num)

        print("Choose Your First Pokemon to battle")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur(death_image = "/Users/zhangyifeng/Desktop/collection-ai/task1/getup-zyh/PokemonGame/Bulbasaur.jpeg")
    pokemon2 = pokemon.PikaChu(death_image = "/Users/zhangyifeng/Desktop/collection-ai/task1/getup-zyh/PokemonGame/Pikachu.jpeg")
    pokemon3 = pokemon.Squirtle(death_image = "/Users/zhangyifeng/Desktop/collection-ai/task1/getup-zyh/PokemonGame/Squirtle.jpeg")
    pokemon4 = pokemon.Charmander(death_image = "/Users/zhangyifeng/Desktop/collection-ai/task1/getup-zyh/PokemonGame/Charmander.jpeg")
    pokemon5 = pokemon.Pidgey(death_image = "/Users/zhangyifeng/Desktop/collection-ai/task1/getup-zyh/PokemonGame/Pidge.png")
    all_pokemon = [pokemon1,pokemon2,pokemon3,pokemon4,pokemon5]
    play = Play(all_pokemon)
    play.run()