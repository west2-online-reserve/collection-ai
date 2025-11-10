from __future__ import annotations
import copy  
import random  
import sys 
import pokemon


def valid_choice(choice, range):
    """
    验证用户输入的选择是否有效
    """
    return choice.isdigit() and 1 <= int(choice) <= range 


class Play:
    def __init__(self, all_pokemon) -> None:
        """
        初始化游戏
        :param all_pokemon: 所有可选择的宝可梦列表
        """
        self.all_pokemon = all_pokemon  # 所有可用的宝可梦
        self.player_team = []  # 玩家队伍
        self.computer_team = []  # 电脑队伍
        self.current_player_pokemon = None  # 玩家当前使用的宝可梦
        self.current_computer_pokemon = None  # 电脑当前使用的宝可梦
        self.turn = 0  # 回合计数器

    def player_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        """
        让玩家选择他们的宝可梦队伍
        :param pokemon_to_choose: 可供选择的宝可梦列表
        :param num: 需要选择的宝可梦数量，默认为3
        """
        print(f"为你的队伍选择 {num} 个宝可梦:")
        # 创建列表副本,避免修改原始列表
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 0
        # 选择了足够数量的宝可梦
        while len(self.player_team) < num:
            # 显示可选择的宝可梦列表
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"请选择 {index} 号宝可梦:")
            # 验证选择是否有效
            if valid_choice(choice, len(pokemon_to_choose)):
                # 将选择的宝可梦深拷贝后添加到队伍中并从可选列表中移除
                selected = pokemon_to_choose.pop(int(choice) - 1)
                self.player_team.append(copy.deepcopy(selected))
                index += 1
            else:
                print("无效选择，请选择一个有效的宝可梦")
        # 将玩家队伍中所有宝可梦的 team 属性设为 "玩家"
        for p in self.player_team:
            try:
                p.team = "玩家"
            except Exception:
                pass
        print("这是你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(self, pokemon_to_choose: list, num=3):
        """
        电脑随机选择宝可梦队伍
        :param pokemon_to_choose: 可供选择的宝可梦列表
        :param num: 需要选择的宝可梦数量，默认为3
        """
        print(f"对手正在选择 {num} 号宝可梦")
        # 随机选择指定数量的宝可梦
        chosen = random.sample(pokemon_to_choose, num)
        # 深拷贝后加入电脑队伍，使对象独立，team 属性不会互相覆盖
        for s in chosen:
            try:
                self.computer_team.append(copy.deepcopy(s))
            except Exception:
                # 如果无法深拷贝则直接加入原实例
                self.computer_team.append(s)
        # 将电脑队伍中所有宝可梦的 team 属性设为 "电脑"
        for p in self.computer_team:
            try:
                p.team = "电脑"
            except Exception:
                pass
        print("这是对手的宝可梦队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list):
        """
        打印宝可梦列表
        :param pokemon_list: 要显示的宝可梦列表
        """
        # 从1开始编号显示每个宝可梦
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def player_choose_pokemon(self):
        """
        让玩家选择当前要参战的宝可梦
        :return: 返回玩家选择的宝可梦
        """
        print("你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("选择参战宝可梦的编号:")
            if valid_choice(choice, len(self.player_team)):
                # 获取选择的宝可梦
                chosen_pokemon = self.player_team[int(choice) - 1]
                # 检查选择的宝可梦是否还能战斗
                if chosen_pokemon.alive is True:
                    print(f"你选择了 {chosen_pokemon.name}")
                    self.current_player_pokemon = chosen_pokemon
                    return chosen_pokemon
                else:
                    print(f"{chosen_pokemon.name} 已经失去战斗能力！请选择其他宝可梦！")
            else:
                print("无效选择，请选择一个有效的宝可梦")

    def computer_choose_pokemon(self):
        """
        电脑随机选择一个能战斗的宝可梦
        """
        # 获取所有还能战斗的宝可梦
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        # 随机选择一个
        chosen_pokemon = random.choice(available_pokemon)
        print(f"对手选择了 {chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon

    def game_finished(self):
        """
        游戏结束，退出程序
        """
        sys.exit()

    def check_game_status(self):

        # 检查双方是否所有宝可梦都失去战斗能力
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)

        # 根据不同情况判断游戏结果
        if is_player_fail and is_computer_fail:

            print("你的宝可梦和对手的宝可梦都已经失去战斗能力，游戏平局。")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:

            print("对手的宝可梦都已经失去战斗能力，你赢了！")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:

            print("你的宝可梦都已经失去战斗能力，你输了！")
            self.game_finished()

        # 如果当前的宝可梦失去战斗能力，需要更换新的宝可梦
        if not self.current_player_pokemon.alive:
            self.player_choose_pokemon()
        if not self.current_computer_pokemon.alive:
            self.computer_choose_pokemon()

    def player_use_skills(self):
        """
        让玩家选择并使用技能
        """
        print("选择你的宝可梦要使用的技能")
        skills = self.current_player_pokemon.skills
        # 显示可用技能列表
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
        choice = input("选择你想要使用的技能编号:")
        # 验证选择是否有效
        if valid_choice(choice, len(skills)):
            player_skill = self.current_player_pokemon.skills[int(choice) - 1]
            # 使用选择的技能攻击对手
            self.current_player_pokemon.use_skill(
                player_skill, self.current_computer_pokemon
            )
        else:
            print("无效选择，请选择一个有效的技能")

    def computer_use_skills(self):
        """
        电脑随机选择并使用技能
        """
        # 从当前宝可梦的技能中随机选择一个
        computer_skill = random.choice(self.current_computer_pokemon.skills)
        # 使用选择的技能攻击玩家的宝可梦
        self.current_computer_pokemon.use_skill(
            computer_skill, self.current_player_pokemon
        )

    def battle_round_begin(self):
        """
        回合开始时的处理
        """
        # 触发双方宝可梦的回合开始效果
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()

        # 在新的回合开始时，先将双方标记为可以行动（状态效果可以在下面将其设为不可行动）
        try:
            self.current_player_pokemon.can_act = True
        except Exception:
            pass
        try:
            self.current_computer_pokemon.can_act = True
        except Exception:
            pass

        # 应用所有状态效果
        self.current_player_pokemon.apply_status_effect()
        self.current_computer_pokemon.apply_status_effect()

        # 检查游戏状态
        self.check_game_status()

    def battle_round(self):
        """
        执行一个完整的战斗回合
        """
        # 显示当前对战的宝可梦
        print(
            f"\n{self.current_player_pokemon.team}的{self.current_player_pokemon.name} vs {self.current_computer_pokemon.team}的{self.current_computer_pokemon.name}"
        )
        # 玩家和电脑依次使用技能
        self.player_use_skills()
        # 检查游戏状态
        self.check_game_status()
        
        self.computer_use_skills()
        # 检查游戏状态
        self.check_game_status()

    def run(self):
        """
        游戏主循环，包含整个游戏的流程
        """
        # 双方选择初始队伍
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        # 选择首发宝可梦
        print("选择你的首发宝可梦进行战斗")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()


        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    pokemon1 = pokemon.Bulbasaur()
    pokemon2 = pokemon.Pikachu()
    pokemon3 = pokemon.Squirtle()
    pokemon4 = pokemon.Charmander()
    pokemon5 = pokemon.Elegant_penguin()
    all_pokemon = [pokemon1, pokemon2, pokemon3, pokemon4, pokemon5]
    play = Play(all_pokemon)
    play.run()
