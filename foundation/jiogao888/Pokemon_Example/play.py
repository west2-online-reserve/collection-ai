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

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        # 玩家选择队伍中的Pokemon
        print(f"Choose {num} pokemon for your team:")
        pokemon_to_choose = copy.deepcopy(pokemon_to_choose)
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"Select your pokemon {index} by number: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                your_pokemon = copy.deepcopy(pokemon_to_choose.pop(int(choice) - 1))
                self.player_team.append(your_pokemon)
                index += 1
            else:
                print("Invalid choice, please select a valid Pokemon")
        for pokemon in self.player_team:
            pokemon.name = "yours " + pokemon.name
        print("Here is your pokemon team:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        # 电脑选择对战的队伍
        print(f"Your opponent is choosing {num} pokemon")
        computer_choicepokemon = copy.deepcopy(all_pokemon)
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        for pokemon in self.computer_team:
            pokemon.name = "computer's " + pokemon.name
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
                if chosen_pokemon.alive is True:
                    print(f"You choosed {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    print(f"{chosen_pokemon.name} has fainted! Choose another Pokemon!")
            else:
                print("Invalid choice, please select a valid Pokemon")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"Your opponent choosed {chosen_pokemon}")
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
            print("All your and opponent's Pokemon have fainted. The game is tied.")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("All computer's Pokemon have fainted. You win!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("All your Pokemon have fainted. You lose!")
            self.game_finished()

        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        # 玩家选择技能
        if self.current_player_pokemon is None:
            return
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
        if self.current_computer_pokemon is None:
            return
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def battle_round_begin(self):
        # 回合开始
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    def battle_round(self):
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        # 回合进行
        print(
            f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}"
        )
        if (
            self.current_player_pokemon.avoid == True
            and self.current_player_pokemon.type == "Elec4tric"
        ):
            print(f"{self.current_player_pokemon.name} move again.")
            self.player_use_skills()
            self.current_player_pokemon.avoid = False
        self.player_use_skills()
        if (
            self.current_computer_pokemon.avoid == True
            and self.current_computer_pokemon.type == "Electric"
        ):
            print(f"{self.current_computer_pokemon.name} move again.")
            self.computer_use_skills()
            self.current_computer_pokemon.avoid == False
        self.computer_use_skills()
        self.check_game_status()

    def run(self):
        # 游戏主循环
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        print("Choose Your First Pokemon to battle")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.Charmander()
    pokemon3 = pokemon.Squirtle()
    pokemon4 = pokemon.PikaChu()
    pokemon5 = pokemon.Geodude()

    all_pokemon = [pokemon1, pokemon2, pokemon3, pokemon4, pokemon5]
    play = Play(all_pokemon)
    play.run()
# if __name__ == "__main__":
#     # ... 原有的初始化代码 ...

#     # 分离测试小火龙
#     print("\n===== 开始小火龙技能隔离测试 =====")
#     my_charmander = pokemon.Charmander()
#     enemy_charmander = pokemon.Charmander()

#     # 直接获取 Ember 技能
#     ember_skill = my_charmander.skills[0]  # 假设第一个是 Ember
#     print(f"准备使用技能: {ember_skill.name}")

#     # 绕过 Play 类，直接调用技能！
#     ember_skill.execute(my_charmander, enemy_charmander)

#     print("===== 技能测试结束 =====")
