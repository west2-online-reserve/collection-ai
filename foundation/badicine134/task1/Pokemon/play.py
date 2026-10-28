from __future__ import annotations
import copy
import random
import sys
import time

from skills import FlameCharge
from effects import ChargeEffect
import pokemon
from weather import Weather


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
        self.weather = Weather()

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        # 玩家选择队伍中的 Pokemon
        print(f"请选择 3 个宝可梦用于组成你的队伍:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 1
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"\n选择你的第 {index} 个宝可梦: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1))
                index += 1
            else:
                print("\n不合法的选择, 请重新选择")
        print("\n你的队伍:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        # 电脑选择对战的队伍
        self.computer_team.extend(random.sample(pokemon_to_choose, num))
        print("\n电脑的队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印 Pokemon 列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}", end=" ")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的 Pokemon
        while True:
            print("\n")
            self.print_pokemon_list(self.player_team)
            choice = input("\n选择出战的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"\n你选择了 {chosen_pokemon.name}！")
                    self.current_player_pokemon = chosen_pokemon
                    self.current_player_pokemon.tag = "你的 "
                    # 重置宝可梦各项数值
                    self.current_player_pokemon.reset()
                    return chosen_pokemon  # 返回选择的 Pokemon
                else:
                    print(f"\n{chosen_pokemon.name} 昏厥了! 选择另一个宝可梦出战!")
            else:
                print("\n不合法的选择, 请重新选择")

    def computer_choose_pokemon(self):  # 电脑的智能系统有待修改
        # 电脑随机选择一个存活的 Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"电脑选择了 {chosen_pokemon.name}！")
        self.current_computer_pokemon = chosen_pokemon
        self.current_computer_pokemon.tag = "电脑的 "
        self.current_computer_pokemon.reset()
        return chosen_pokemon  # 返回选择的 Pokemon

    def game_finished(self):
        # 游戏结束
        sys.exit()

    def check_game_status(self):
        # 检查游戏状态，判断玩家或电脑是否失败
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("\n你和对手的所有宝可梦都昏厥了。 平局。")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("\n你对手的所有宝可梦都昏厥了。 你赢了！")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("\n你的所有宝可梦都昏厥了。 你输了！")
            self.game_finished()

        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        if self.current_player_pokemon is None:
            return
        # 前一回合使用FlameCharge：这回合不能选技能，结算完其他东西后直接发射
        if any(isinstance(status, ChargeEffect) for status in self.current_player_pokemon.statuses[:]):
            flame_skill = next((s for s in self.current_player_pokemon.skills if isinstance(s, FlameCharge)),None)
            if flame_skill:
                result = self.current_player_pokemon.use_skill(
                    flame_skill, self.current_computer_pokemon, self.weather
                )
                # 如果被麻痹，移除蓄力状态
                if result is None:
                    for s in self.current_player_pokemon.statuses[:]:
                        if isinstance(s, ChargeEffect):
                            self.current_player_pokemon.statuses.remove(s)
                            print(f"【状态提示】由于麻痹状态，{self.current_player_pokemon.name} 的蓄力被打断了！")
                return
        # 玩家选择技能
        print(f"\n{self.current_player_pokemon} 的技能：")
        skills = self.current_player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        while True:
            choice = input("\n选择要使用的技能:")
            if valid_choice(choice, len(skills)):
                player_skill = self.current_player_pokemon.skills[int(choice) - 1]
                result = self.current_player_pokemon.use_skill(
                    player_skill, self.current_computer_pokemon, self.weather
                )
                # 电属性宝可梦特性
                if result == "electric_dodge":
                    computer_skill = random.choice(self.current_computer_pokemon.skills)
                    self.current_computer_pokemon.use_skill(
                        computer_skill, self.current_player_pokemon, self.weather
                    )
                return
            else:
                print("\n不合法的选择, 请重新选择")

    def computer_use_skills(self):
        if self.current_computer_pokemon is None:
            return
        # FlameCharge逻辑同上
        if any(isinstance(status, ChargeEffect) for status in self.current_computer_pokemon.statuses[:]):
            flame_skill = next((s for s in self.current_computer_pokemon.skills if isinstance(s, FlameCharge)),None)
            if flame_skill:
                result = self.current_computer_pokemon.use_skill(
                    flame_skill, self.current_player_pokemon, self.weather
                )
                # 如果被麻痹，移除蓄力状态
                if result is None:
                    for s in self.current_computer_pokemon.statuses[:]:
                        if isinstance(s, ChargeEffect):
                            self.current_computer_pokemon.statuses.remove(s)
                            print(f"【状态提示】由于麻痹状态，{self.current_computer_pokemon.name} 的蓄力被打断了！")
                return
        # 电脑随机选择技能
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        result = self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon, self.weather
        )
        # 电属性宝可梦特性
        if result == "electric_dodge":
            skills = self.current_player_pokemon.skills
            for i, skill in enumerate(skills, 1):
                print(f"{i}: {skill}")

            while True:
                choice = input("\n输入要反击的技能:")
                if valid_choice(choice, len(skills)):
                    counter_skill = skills[int(choice) - 1]
                    self.current_player_pokemon.use_skill(
                        counter_skill, self.current_computer_pokemon, self.weather
                    )
                    break
                else:
                    print("\n不合法的选择, 请重新选择")

    def battle_round_begin(self):
        # 回合开始
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        time.sleep(1)
        print('\n==================== 新的回合 ====================')
        # 更新天气
        self.weather.update_weather()

        self.current_player_pokemon.begin(self.weather)
        self.current_player_pokemon.apply_status_effect()
        self.current_computer_pokemon.begin(self.weather)
        self.current_computer_pokemon.apply_status_effect()
        self.check_game_status()

    def battle_round(self):
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        # 回合进行
        # 增加用速度来判定出手顺序
        if self.current_player_pokemon.speed >= self.current_computer_pokemon.speed:
            print(f"\n{self.current_player_pokemon.name} 速度更快，先发动攻击！")
            self.player_use_skills()
            if self.current_computer_pokemon.alive:
                self.computer_use_skills()
        else:
            print(f"\n{self.current_computer_pokemon.name} 速度更快，先发动攻击！")
            self.computer_use_skills()
            if self.current_player_pokemon.alive:
                self.player_use_skills()

        self.check_game_status()

    def run(self):
        # 游戏主循环
        player_pool = copy.deepcopy(self.all_pokemon)  # 深拷贝 由ai教学
        computer_pool = copy.deepcopy(self.all_pokemon)

        self.player_choose_pokemon_team(player_pool)
        self.computer_choose_pokemon_team(computer_pool)

        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    all_pokemon = [
        pokemon.Bulbasaur(),
        pokemon.Bellsprout(),
        pokemon.Squirtle(),
        pokemon.Psyduck(),
        pokemon.Charmander(),
        pokemon.Vulpix(),
        pokemon.Pikachu(),
        pokemon.Magnemite()
    ]
    play = Play(all_pokemon)
    play.run()
