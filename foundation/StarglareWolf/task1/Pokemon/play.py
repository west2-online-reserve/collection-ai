from __future__ import annotations
import copy
import random
import sys
import time

import pokemon
from color import *

def valid_choice(choice, range):
    # 判断输入的选择是否合法
    return choice.isdigit() and 1 <= int(choice) <= range


class Play:
    def __init__(self, user_pokemon, computer_pokemon) -> None:
        self.user_pokemon = user_pokemon
        self.computer_pokemon = computer_pokemon
        self.player_team = []
        self.computer_team = []
        self.current_player_pokemon = None
        self.current_computer_pokemon = None
        self.turn = 0

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num= 3):
        # 玩家选择队伍中的 Pokemon
        pokemon_to_choose = copy.copy(pokemon_to_choose)       # 浅拷贝，复制一份，防止修改原列表
        self.print_pokemon_list(pokemon_to_choose)
        print(f"为你的队伍选择最多 {num} 个宝可梦: ", end = "")
        while True:
            choice = list(map(int, input().split()))
            length1 = len(choice)
            if length1 > num:
                print(f"最多选择 {num} 个宝可梦: ", end = "")
            elif length1 == 0:
                print("至少选择一个宝可梦: ",end = "")
            else:
                choice = list(filter(lambda x: 0 < x <= len(pokemon_to_choose), choice))
                length2 = len(choice)
                if length1 != length2:
                    print("存在无效的输入,请重新选择:", end=" ")
                else:
                    se = set(choice)
                    if len(se) != len(choice):
                        print("重复选择宝可梦,请重新选择:", end=" ")
                    else:
                        for num in choice:
                            self.player_team.append(pokemon_to_choose[num - 1])
                        break

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num= 3):
        # 电脑选择对战的队伍
        print(f"\n你的敌人正在选择 {num} 个宝可梦……")
        self.computer_team.extend(random.sample(pokemon_to_choose, num))  # extend: list 方法，原地修改列表，将新东西追加到列表末尾； sample： random 方法，随机抽出 num 个元素，合成一个列表返回，而 extend 可以把列表拆开取元素
        print("敌人的宝可梦队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        # 打印 Pokemon 列表
        for i, p in enumerate(pokemon_list, 1):     # 从序号 1 开始一直迭代，同时迭代列表中的所有元素，就不用 i += 1 了
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        # 玩家选择当前战斗的 Pokemon
        print("\n你的队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("\n选择你的宝可梦出战:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"你选择了 {chosen_pokemon.name} 属性: {chosen_pokemon.type}")
                    self.current_player_pokemon = chosen_pokemon
                    self.current_player_pokemon.distinguish_name(PLAYER, OPPONENT)
                    return chosen_pokemon  # 返回选择的 Pokemon
                else:
                    print(f"{chosen_pokemon.name} 已昏厥! 重新选一个!")
            else:
                print("无效的选择，请选择有效的宝可梦编号")

    def computer_choose_pokemon(self):
        # 电脑随机选择一个存活的 Pokemon
        available_pokemon = [p for p in self.computer_team if p.alive is True]    # 又复制一份列表，得到活着的宝可梦
        chosen_pokemon = random.choice(available_pokemon)                         # 这样简洁了许多
        print(f"你的对手选择了 {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        self.current_computer_pokemon.distinguish_name(PLAYER, OPPONENT)
        return chosen_pokemon  # 返回选择的 Pokemon

    def game_finished(self):
        # 游戏结束
        sys.exit()

    def check_game_status(self):
        # 检查游戏状态，判断玩家或电脑是否失败
        # 这里有个坑，就是主函数中总是在双方释放技能后才 check, 导致明明某一方的宝可梦已经昏厥了，仍然能释放技能弄你, 所以必须在释放技能前先单独判断双方当前宝可梦状态
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)      # all : 内置函数，迭代，判断所有条件是否满足
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("\n双方的宝可梦全部昏厥,平局!")
            self.game_finished()
        elif is_computer_fail:
            print("\n电脑的宝可梦全部昏厥，你胜利!")
            self.game_finished()
        elif is_player_fail:
            print("\n你的宝可梦全部昏厥，你失败!")
            self.game_finished()

        # 宝可梦替换规则：当前宝可梦晕厥方可替换
        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        # 玩家选择技能
        if self.current_player_pokemon is None:
            return
        if self.current_player_pokemon.alive:
            if not self.current_player_pokemon.paralyzed:    # 没有麻痹
                print("\n你的宝可梦可使用技能如下")
                skills = self.current_player_pokemon.skills
                for i, skill in enumerate(skills, 1):
                    print(f"{i}: {skill}")
                while True:
                    choice = input("\n选择一个技能释放:")
                    if valid_choice(choice, len(skills)):
                        self.current_player_pokemon.current_skill = self.current_player_pokemon.skills[int(choice) - 1]
                        if self.current_player_pokemon.use_skill(
                            self.current_player_pokemon.current_skill, self.current_computer_pokemon
                        ):
                            break
                    else:
                        print("无效的选择，请选择有效的技能编号")
                time.sleep(0.2)
            else:
                print(f"{self.current_computer_pokemon.master}的 {self.current_player_pokemon.name} 处于麻痹状态,无法发动攻击")


    def computer_use_skills(self):
        # 电脑随机选择技能
        if self.current_computer_pokemon is None:
            return
        if self.current_computer_pokemon.alive:
            if not self.current_computer_pokemon.paralyzed:
                self.current_computer_pokemon.current_skill = random.choice(self.current_computer_pokemon.skills)

                while not self.current_computer_pokemon.use_skill(
                    self.current_computer_pokemon.current_skill, self.current_player_pokemon
                ):
                    self.current_computer_pokemon.current_skill = random.choice(self.current_computer_pokemon.skills)
            else:
                print(f"{self.current_computer_pokemon.master}的 {self.current_computer_pokemon} 处于麻痹状态,无法发动攻击")

    def battle_round_begin(self):
        # 回合开始
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        time.sleep(0.2)
        print("\n-------回合开始-------")
        self.current_player_pokemon.begin(self.current_computer_pokemon)
        self.current_computer_pokemon.begin(self.current_player_pokemon)
        self.check_game_status()

    def battle_round(self):
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        # 回合进行
        print(
            f"{self.current_player_pokemon.name}[{self.current_player_pokemon.hp} / {self.current_player_pokemon.max_hp}]"
            f" vs {self.current_computer_pokemon.name}[{self.current_computer_pokemon.hp} / {self.current_computer_pokemon.max_hp}]"
        )
        self.player_use_skills()
        self.computer_use_skills()
        self.check_game_status()

    def run(self):
        # 游戏主循环
        self.player_choose_pokemon_team(self.user_pokemon)
        self.computer_choose_pokemon_team(self.computer_pokemon)

        print("选择宝可梦进行战斗!")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    # 显式，同时单独在列表里创建对象，防止两个列表“数据共享”，这样也是为了对抗时能分清楚谁是谁
    player = POSITION["player"]
    enemy = POSITION["opponent"]
    user_pokemon = [pokemon.Bulbasaur(player), pokemon.PikaChu(player), pokemon.Squirtle(player), pokemon.Charmander(player)]
    computer_pokemon = [pokemon.Bulbasaur(enemy), pokemon.PikaChu(enemy), pokemon.Squirtle(enemy), pokemon.Charmander(enemy)]
    play = Play(user_pokemon, computer_pokemon)
    play.run()