from __future__ import annotations
import copy
import random
import sys
import pokemons
from misc.tools import printWithDelay
from settings import POKEMON_NUM


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

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=POKEMON_NUM):
        # 玩家选择队伍中的Pokemon
        printWithDelay(f"为你的队伍选择 {num} 个宝可梦:")
        pokemon_to_choose = copy.deepcopy(pokemon_to_choose)
        index = 1
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"选择你的第 {index} 个宝可梦: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                printWithDelay("非法的选择,请正确选择宝可梦")
        printWithDelay("这是你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=POKEMON_NUM):
        # 电脑选择对战的队伍
        printWithDelay(f"你的对手正在选择 {num} 个宝可梦")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        printWithDelay("这是你对手的队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的Pokemon
        printWithDelay("你的队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("输入你要选择的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    printWithDelay(f"你选择了 {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    printWithDelay(f"{chosen_pokemon.name} 倒下了! 请选择其他的宝可梦!")
            else:
                printWithDelay("非法的选择,请正确选择宝可梦")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        printWithDelay(f"你的对手选择了 {chosen_pokemon}")
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
            printWithDelay("你和你对手的所有宝可梦都倒下了. 平局.")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            printWithDelay("你对手的所有宝可梦都倒下了. 你赢了!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            printWithDelay("你所有的宝可梦都倒下了. 你输了!")
            self.game_finished()
        if not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        # 玩家选择技能
        printWithDelay("选择技能")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        while True:
            choice = input("选择你要使用的技能:")
            if valid_choice(choice, len(skills)):
                player_skill = self.current_player_pokemon.skills[int(choice) - 1]
                self.current_player_pokemon.use_skill(
                    player_skill, self.current_computer_pokemon
                )
                return
            else:
                printWithDelay("非法选择,请选择正确的技能")

    def computer_use_skills(self):
        # 电脑随机选择技能
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def player_action(self):
        printWithDelay("~" * 10, "行动", "~" * 10)

        self.player_use_skills()
        self.check_game_status()
        if (
            self.current_computer_pokemon.type == "电"
            and self.current_computer_pokemon.is_dodged
        ):
            self.current_computer_pokemon.is_dodged = False
            printWithDelay(f"对手的 {self.current_computer_pokemon.name} 获得额外回合")
            self.computer_action()

    def computer_action(self):
        printWithDelay("~" * 10, "行动", "~" * 10)
        self.computer_use_skills()
        self.check_game_status()
        if (
            self.current_player_pokemon.type == "电"
            and self.current_player_pokemon.is_dodged
        ):
            self.current_player_pokemon.is_dodged = False
            printWithDelay(f"你的 {self.current_player_pokemon.name} 获得额外回合")
            self.player_action()

    def battle_round(self):
        # 回合进行
        printWithDelay(
            f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}"
        )
        printWithDelay("=" * 15, "你的回合", "=" * 15)
        printWithDelay("~" * 10, "效果结算", "~" * 10)
        self.current_player_pokemon.begin()
        self.check_game_status()
        if not self.current_player_pokemon.cant_move:
            self.player_action()
        printWithDelay("=" * 15, "对手回合", "=" * 15)
        printWithDelay("~" * 10, "效果结算", "~" * 10)
        self.current_computer_pokemon.begin()
        self.check_game_status()
        if not self.current_computer_pokemon.cant_move:
            self.computer_action()

    def run(self):
        # 游戏主循环
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)
        printWithDelay("选择你第一个作战的宝可梦")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemons.Bulbasaur()
    pokemon2 = pokemons.PikaChu()
    pokemon3 = pokemons.Squirtle()
    pokemon4 = pokemons.Charmander()
    pokemon5 = pokemons.Ditto()
    all_pokemon = [pokemon1, pokemon2, pokemon3, pokemon4, pokemon5]
    play = Play(all_pokemon)
    play.run()
