"""宝可梦命令行对战：入口 python play.py。"""

from __future__ import annotations

import copy
import random
import re

from pokemon import POKEDEX, Pokemon


def valid_choice(choice: str, maximum: int) -> bool:
    return choice.isdigit() and 1 <= int(choice) <= maximum


class Play:
    def __init__(self, all_pokemon: list[Pokemon]) -> None:
        self.all_pokemon = all_pokemon
        self.player_team: list[Pokemon] = []
        self.computer_team: list[Pokemon] = []
        self.current_player_pokemon: Pokemon | None = None
        self.current_computer_pokemon: Pokemon | None = None
        self.finished = False

    def player_choose_pokemon_team(
        self, pokemon_to_choose: list[Pokemon], num: int = 3
    ) -> None:
        print(f"请选择 {num} 个宝可梦用于组成你的队伍：")
        self.print_pokemon_list(pokemon_to_choose)

        while True:
            raw = input("输入数字选择你的宝可梦：").strip()
            parts = [part for part in re.split(r"[\s,，、]+", raw) if part]
            if len(parts) != num or not all(part.isdigit() for part in parts):
                print(f"请输入 {num} 个数字，例如：1 2 4。")
                continue
            numbers = [int(part) for part in parts]
            if any(number < 1 or number > len(pokemon_to_choose) for number in numbers):
                print(f"数字必须在 1 到 {len(pokemon_to_choose)} 之间。")
                continue
            if len(set(numbers)) != num:
                print("不能重复选择同一只宝可梦。")
                continue
            self.player_team = [
                copy.deepcopy(pokemon_to_choose[number - 1]) for number in numbers
            ]
            return

    def computer_choose_pokemon_team(
        self, pokemon_to_choose: list[Pokemon], num: int = 3
    ) -> None:
        self.computer_team = [
            copy.deepcopy(pokemon)
            for pokemon in random.sample(pokemon_to_choose, num)
        ]

    def print_pokemon_list(self, pokemon_list: list[Pokemon]) -> None:
        print("  ".join(f"{i}. {pokemon}" for i, pokemon in enumerate(pokemon_list, 1)))

    def player_choose_pokemon(self) -> Pokemon:
        print("\n请选择你的宝可梦：")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("输入数字选择你的宝可梦：")
            if valid_choice(choice, len(self.player_team)):
                chosen = self.player_team[int(choice) - 1]
                if chosen.alive:
                    print(f"你选择了 {chosen.name}！")
                    self.current_player_pokemon = chosen
                    return chosen
                print(f"{chosen.name} 已经倒下了，请选择另一只。")
            else:
                print("输入无效，请重新选择。")

    def computer_choose_pokemon(self) -> Pokemon:
        available = [pokemon for pokemon in self.computer_team if pokemon.alive]
        chosen = random.choice(available)
        print(f"电脑选择了 {chosen.name}！")
        self.current_computer_pokemon = chosen
        return chosen

    def game_finished(self) -> None:
        self.finished = True

    def check_game_status(self) -> None:
        player_fail = all(not pokemon.alive for pokemon in self.player_team)
        computer_fail = all(not pokemon.alive for pokemon in self.computer_team)

        if player_fail and computer_fail:
            print("双方宝可梦都倒下了，平局！")
            self.game_finished()
            return
        if computer_fail:
            print("电脑的宝可梦都倒下了，你赢了！")
            self.game_finished()
            return
        if player_fail:
            print("你的宝可梦都倒下了，你输了！")
            self.game_finished()
            return

        if (
            self.current_player_pokemon is not None
            and not self.current_player_pokemon.alive
        ):
            self.current_player_pokemon = self.player_choose_pokemon()
        if (
            self.current_computer_pokemon is not None
            and not self.current_computer_pokemon.alive
        ):
            self.current_computer_pokemon = self.computer_choose_pokemon()

    def player_use_skills(self) -> None:
        player = self.current_player_pokemon
        computer = self.current_computer_pokemon
        if player is None or computer is None or not player.alive:
            return

        if player.skip_action_reason is not None:
            print(f"{player.name} 因「{player.skip_action_reason}」无法行动！")
            player.skip_action_reason = None
            return

        if player.charging_skill is not None:
            player.charging_skill.execute(player, computer)
            return

        print(f"你的 {player.name} 的技能：")
        for i, skill in enumerate(player.skills, 1):
            print(f"{i}. {skill}")
        choice = input("选择一个技能进行攻击：")
        if valid_choice(choice, len(player.skills)):
            player.skills[int(choice) - 1].execute(player, computer)
        else:
            print("输入无效，请重新选择。")

    def computer_use_skills(self) -> None:
        computer = self.current_computer_pokemon
        player = self.current_player_pokemon
        if computer is None or player is None or not computer.alive:
            return

        if computer.skip_action_reason is not None:
            print(f"{computer.name} 因「{computer.skip_action_reason}」无法行动！")
            computer.skip_action_reason = None
            return

        if computer.charging_skill is not None:
            computer.charging_skill.execute(computer, player)
            return

        random.choice(computer.skills).execute(computer, player)

    def battle_round_begin(self) -> None:
        player = self.current_player_pokemon
        computer = self.current_computer_pokemon
        if player is None or computer is None:
            return

        player.skip_action_reason = None
        computer.skip_action_reason = None
        player.begin()
        computer.begin()
        self.check_game_status()

    def battle_round(self) -> None:
        player = self.current_player_pokemon
        computer = self.current_computer_pokemon
        if player is None or computer is None:
            return

        self.player_use_skills()
        print()
        self.computer_use_skills()
        self.check_game_status()

    def run(self) -> None:
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        print("\n你的队伍：")
        self.print_pokemon_list(self.player_team)
        print("电脑的队伍：")
        self.print_pokemon_list(self.computer_team)

        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while not self.finished:
            self.battle_round_begin()
            if self.finished:
                break
            self.battle_round()


def main() -> None:
    game = Play(list(POKEDEX))
    try:
        game.run()
    except (EOFError, KeyboardInterrupt):
        print("\n游戏结束。")


if __name__ == "__main__":
    main()