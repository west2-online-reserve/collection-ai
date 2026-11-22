from __future__ import annotations
import copy
import random
import sys
import time
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
        # 玩家选择队伍中的Pokemon（支持逐个输入，也支持空格分隔一次输入多个数字）
        time.sleep(0.3)
        print(f"请选择 {num} 个宝可梦用于组成你的队伍：")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            raw = input("输入数字选择你的宝可梦：").strip()
            parts = raw.split()
            if len(parts) > 1:
                needed = num - len(self.player_team)
                if (
                    len(parts) == needed
                    and all(valid_choice(p, len(pokemon_to_choose)) for p in parts)
                    and len(set(parts)) == len(parts)
                ):
                    indices = sorted([int(p) - 1 for p in parts], reverse=True)
                    chosen = [copy.deepcopy(pokemon_to_choose[int(p) - 1]) for p in parts]
                    for idx in indices:
                        pokemon_to_choose.pop(idx)
                    self.player_team.extend(chosen)
                    break
                else:
                    time.sleep(0.3)
                    print("选择无效，请输入正确的宝可梦编号！")
            elif valid_choice(raw, len(pokemon_to_choose)):
                self.player_team.append(copy.deepcopy(pokemon_to_choose.pop(int(raw) - 1)))
            else:
                time.sleep(0.3)
                print("选择无效，请输入正确的宝可梦编号！")
        time.sleep(0.4)
        print("\n你的宝可梦队伍：")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        # 电脑选择对战的队伍
        time.sleep(0.5)
        print(f"\n电脑正在选择 {num} 个宝可梦...")
        self.computer_team.extend([copy.deepcopy(p) for p in random.sample(pokemon_to_choose, num)])
        time.sleep(0.6)
        print("电脑的宝可梦队伍：")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印Pokemon列表
        time.sleep(0.2)
        print(" ".join(f"{i}. {p}" for i, p in enumerate(pokemon_list, 1)))

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的Pokemon
        time.sleep(0.4)
        print("\n请选择你的宝可梦：")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("输入数字选择你的宝可梦：").strip()
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    time.sleep(0.4)
                    print(f"你选择了 {chosen_pokemon.name}！")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon
                else:
                    time.sleep(0.3)
                    print(f"{chosen_pokemon.name} 已经昏厥，请选择其他宝可梦！")
            else:
                time.sleep(0.3)
                print("选择无效，请输入正确的宝可梦编号！")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        time.sleep(0.6)
        print(f"电脑选择了 {chosen_pokemon.name}！")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon

    def game_finished(self):
        # 游戏结束
        sys.exit()

    def check_game_status(self):
        # 检查游戏状态，判断玩家或电脑是否失败
        is_player_fail = all(p.alive is False for p in self.player_team)
        is_computer_fail = all(p.alive is False for p in self.computer_team)
        if is_player_fail and is_computer_fail:
            time.sleep(0.8)
            print("双方所有宝可梦均已昏厥，平局！")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            time.sleep(0.8)
            print("电脑的所有宝可梦均已昏厥，你赢了！")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            time.sleep(0.8)
            print("你的所有宝可梦均已昏厥，你输了！")
            self.game_finished()

        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        # 玩家选择技能
        if self.current_player_pokemon is None:
            return
        if self.current_player_pokemon.is_paralyzed:
            time.sleep(0.5)
            print(f"{self.current_player_pokemon.name} 处于麻痹状态，本回合无法行动！")
            return
        if self.current_player_pokemon.charging_skill is not None:
            self.current_player_pokemon.use_skill(
                self.current_player_pokemon.charging_skill, self.current_computer_pokemon
            )
            return
        time.sleep(0.3)
        print(f"\n你的 {self.current_player_pokemon.name} 的技能：")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill}")
        while True:
            choice = input("选择一个技能进行攻击：").strip()
            if valid_choice(choice, len(skills)):
                player_skill = self.current_player_pokemon.skills[int(choice) - 1]
                self.current_player_pokemon.use_skill(
                    player_skill, self.current_computer_pokemon
                )
                break
            else:
                time.sleep(0.3)
                print("选择无效，请输入正确的技能编号！")

    def computer_use_skills(self):
        # 电脑随机选择技能
        if self.current_computer_pokemon is None:
            return
        if self.current_computer_pokemon.is_paralyzed:
            time.sleep(0.5)
            print(f"{self.current_computer_pokemon.name} 处于麻痹状态，本回合无法行动！")
            return
        if self.current_computer_pokemon.charging_skill is not None:
            self.current_computer_pokemon.use_skill(
                self.current_computer_pokemon.charging_skill, self.current_player_pokemon
            )
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
        time.sleep(0.6)
        print(
            f"\n【{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}】"
        )
        comp_before = self.current_computer_pokemon
        self.player_use_skills()
        self.check_game_status()
        if not comp_before.alive:
            return
        self.computer_use_skills()
        self.check_game_status()

    def run(self):
        # 游戏主循环
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    all_pokemon = [
        pokemon.Pikachu(),
        pokemon.Bulbasaur(),
        pokemon.Squirtle(),
        pokemon.Charmander(),
        pokemon.Naiwa()
    ]
    play = Play(all_pokemon)
    play.run()
