from __future__ import annotations
import copy
import random
import sys
from typing import NoReturn

import pokemon


def valid_choice(choice: str, max_choice: int) -> bool:
    # 判断输入的选择是否合法
    if not choice.isdigit():
        return False
    try:  # 保护整数转换
        choice_number = int(choice)
    except ValueError:
        return False
    return 1 <= choice_number <= max_choice


class Play:
    def __init__(self, all_pokemon: list[pokemon.Pokemon]) -> None:
        self.all_pokemon: list[pokemon.Pokemon] = all_pokemon
        self.player_team: list[pokemon.Pokemon] = []
        self.computer_team: list[pokemon.Pokemon] = []
        self.current_player_pokemon: pokemon.Pokemon | None = None
        self.current_computer_pokemon: pokemon.Pokemon | None = None
        self.turn: int = 0

    def player_choose_pokemon_team(
        self, pokemon_to_choose: list[pokemon.Pokemon], num: int = 3
    ) -> None:
        # 玩家选择队伍中的Pokemon
        print(f"Choose {num} pokemon for your team:")
        pokemon_to_choose = copy.deepcopy(
            pokemon_to_choose
        )  # 改为使用deepcopy，避免修改原始队列
        index = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"Select your pokemon {index} by number: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("Invalid choice, please select a valid Pokemon")
        print("Here is your pokemon team:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(
        self, pokemon_to_choose: list[pokemon.Pokemon], num: int = 3
    ) -> None:
        # 电脑选择对战的队伍
        print(f"Your opponent is choosing {num} pokemon")
        pokemon_to_choose = copy.deepcopy(
            pokemon_to_choose
        )  # 添加deepcopy，避免修改原始队列
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("Here is your opponent's team:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list: list[pokemon.Pokemon]) -> None:
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self) -> pokemon.Pokemon:
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

    def computer_choose_pokemon(self) -> pokemon.Pokemon:
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"Your opponent choosed {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon  # 返回选择的Pokemon

    def game_finished(self) -> NoReturn:
        # 游戏结束
        sys.exit()

    def check_game_status(self) -> None:
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

    def player_use_skills(self) -> None:
        # 玩家选择技能
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        if not self.current_player_pokemon.can_act():
            return
        print("Choose the skill your pokemon to use")
        skills = self.current_player_pokemon.skills
        while True:  # 增加循环判断，选择有效才能退出
            for i, skill in enumerate(skills, 1):
                print(f"{i}: {skill}")
            choice = input("Select the skill you want to use by number:")
            if valid_choice(choice, len(skills)):
                player_skill = self.current_player_pokemon.skills[int(choice) - 1]
                self.current_player_pokemon.use_skill(
                    player_skill, self.current_computer_pokemon
                )
                break
            else:
                print("Invalid choice, please select a valid Skill")

    def computer_use_skills(self) -> None:
        # 电脑随机选择技能
        if self.current_computer_pokemon is None or self.current_player_pokemon is None:
            return
        if not self.current_computer_pokemon.can_act():
            return
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def process_extra_action(self, is_player: bool) -> None:
        computer_pokemon = self.current_computer_pokemon
        player_pokemon = self.current_player_pokemon
        if computer_pokemon is None or player_pokemon is None:
            return
        player_pending_before = player_pokemon.extra_action_pending
        computer_pending_before = computer_pokemon.extra_action_pending
        # 处理额外行动机会
        if is_player:
            if player_pokemon.extra_action_pending:
                player_pokemon.extra_action_pending = False
            else:
                return
        else:
            if computer_pokemon.extra_action_pending:
                computer_pokemon.extra_action_pending = False
            else:
                return
        if computer_pokemon.alive and player_pokemon.alive:
            if is_player:
                self.player_use_skills()
            else:
                self.computer_use_skills()
        if is_player:
            player_pokemon.extra_action_pending = False
            computer_pokemon.extra_action_pending = (
                computer_pending_before and computer_pokemon.alive
            )
        else:
            computer_pokemon.extra_action_pending = False
            player_pokemon.extra_action_pending = (
                player_pending_before and player_pokemon.alive
            )
        self.check_game_status()  # 检查游戏状态，判断玩家或电脑是否失败

    def battle_round_begin(self) -> None:
        # 回合开始
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.check_game_status()

    # 新增回合结束结算状态阶段
    def battle_round_end(self) -> None:
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.apply_status_effect()
        self.current_computer_pokemon.apply_status_effect()
        self.check_game_status()

    def battle_round(self) -> None:
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        # 回合进行
        print(
            f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}"
        )
        computer_pokemon = self.current_computer_pokemon
        self.player_use_skills()
        self.check_game_status()
        self.process_extra_action(False)  # 处理闪避后的立即行动
        self.process_extra_action(True)  # 处理电光一闪类使用技能后触发追击的行动
        if computer_pokemon.alive:
            self.computer_use_skills()
            self.check_game_status()
            self.process_extra_action(True)
            self.process_extra_action(False)

    def run(self) -> NoReturn:
        # 游戏主循环
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        print("Choose Your First Pokemon to battle")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()
            self.battle_round_end()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.Squirtle()
    pokemon3 = pokemon.Charmander()
    pokemon4 = pokemon.Pikachu()
    pokemon5 = pokemon.Meowth()
    all_pokemon: list[pokemon.Pokemon] = [
        pokemon1, pokemon2, pokemon3, pokemon4, pokemon5
    ]
    play = Play(all_pokemon)
    play.run()
