"""对战流程控制（Battle Controller）。

Play 负责：组队选人、出战选宠、回合推进、状态检查与胜负判定。
"""
from __future__ import annotations

import copy
import random
import sys
from typing import List

import pokemon
from pokemon import Pokemon


def valid_choice(choice: str, length: int) -> bool:
    """判断输入的选择是否合法（1~length 的数字）。"""
    return choice.isdigit() and 1 <= int(choice) <= length


class Play:
    def __init__(self, all_pokemon: List[Pokemon]):
        self.all_pokemon: List[Pokemon] = all_pokemon
        self.player_team: List[Pokemon] = []
        self.computer_team: List[Pokemon] = []
        self.current_player_pokemon: Pokemon | None = None
        self.current_computer_pokemon: Pokemon | None = None
        self.turn: int = 0

    # ---------------- 组队 ----------------
    def player_choose_pokemon_team(self, pokemon_to_choose: List[Pokemon], num: int = 3):
        """玩家选择队伍中的 Pokemon。"""
        print(f"Choose {num} pokemon for your team:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"Select your pokemon {index + 1} by number: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("Invalid choice, please select a valid Pokemon")
        print("Here is your pokemon team:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: List[Pokemon], num: int = 3):
        """电脑随机选择对战队伍。"""
        print(f"\nYour opponent is choosing {num} pokemon")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("Here is your opponent's team:")
        self.print_pokemon_list(self.computer_team)

    @staticmethod
    def print_pokemon_list(pokemon_list: List[Pokemon]):
        """打印 Pokemon 列表。"""
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    # ---------------- 出战选宠 ----------------
    def player_choose_pokemon(self) -> Pokemon:
        """玩家选择当前战斗的存活 Pokemon。"""
        print("Your Team:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("Select your pokemon to battle by number: ")
            if valid_choice(choice, len(self.player_team)):
                chosen = self.player_team[int(choice) - 1]
                if chosen.alive:
                    print(f"You chose {chosen.name}")
                    self.current_player_pokemon = chosen
                    return chosen
                print(f"{chosen.name} has fainted! Choose another Pokemon!")
            else:
                print("Invalid choice, please select a valid Pokemon")

    def computer_choose_pokemon(self) -> Pokemon:
        """电脑随机选择一个存活的 Pokemon。"""
        available = [p for p in self.computer_team if p.alive]
        chosen = random.choice(available)
        print(f"Your opponent chose {chosen.name}")
        self.current_computer_pokemon = chosen
        return chosen

    # ---------------- 回合与状态 ----------------
    @staticmethod
    def game_finished():
        """结束游戏。"""
        sys.exit(0)

    def check_game_status(self):
        """检查胜负；若出战宠昏厥则重新选宠。"""
        is_player_fail = all(p.alive is False for p in self.player_team)
        is_computer_fail = all(p.alive is False for p in self.computer_team)
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

    # ---------------- 技能使用 ----------------
    def player_use_skills(self):
        """玩家选择技能。"""
        if self.current_player_pokemon is None:
            return
        print("Choose the skill your pokemon uses:")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        choice = input("Select the skill you want to use by number: ")
        if valid_choice(choice, len(skills)):
            self.current_player_pokemon.use_skill(skills[int(choice) - 1], self.current_computer_pokemon)
        else:
            print("Invalid choice, please select a valid Skill")

    def computer_use_skills(self):
        """电脑随机选择技能。"""
        if self.current_computer_pokemon is None:
            return
        skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(skill, self.current_player_pokemon)

    # ---------------- 回合推进 ----------------
    def battle_round_begin(self):
        """回合开始：双方结算状态效果与属性被动。"""
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    def battle_round(self):
        """回合进行：玩家先手，随后电脑行动。"""
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        print(f"\n--- Turn {self.turn + 1}: {self.current_player_pokemon.name} vs {self.current_computer_pokemon.name} ---")
        self.turn += 1
        self.player_use_skills()
        self.check_game_status()
        self.computer_use_skills()
        self.check_game_status()

    def run(self):
        """游戏主循环。"""
        self.player_choose_pokemon_team(self.all_pokemon, num=3)
        self.computer_choose_pokemon_team(self.all_pokemon, num=3)
        print("\nChoose Your First Pokemon to battle")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()
        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    all_pokemon = [
        pokemon.Bulbasaur(),
        pokemon.PikaChu(),
        pokemon.Squirtle(),
        pokemon.Charmander(),
        pokemon.Vulpix(),
    ]
    play = Play(all_pokemon)
    play.run()
