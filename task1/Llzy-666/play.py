from time import sleep
import copy
import random
import sys

import pokemon

# 判断输入的选择是否合法
def valid_choice(choice, range):
    return choice.isdigit() and 1 <= int(choice) <= range

# 游戏类
class Play():
    def __init__(self, all_pokemon):
        self.all_pokemon = all_pokemon
        self.player_team = []
        self.computer_team = []
        self.player_pokemon = None
        self.computer_pokemon = None

# 玩家选择宝可梦队伍
    def player_choose_team(self, pokemon_to_choose: list, num=3):
        #两个参数:一个指定为列表,另一个为选择的宝可梦数量
        print(f"选择{num}个宝可梦来组成你的队伍:")
        pokemon_to_choose = copy.deepcopy(pokemon_to_choose)
        idx = 0
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"输入对应的数字来选取你的第{idx + 1}个宝可梦: ")
            print()
            if valid_choice(choice, len(pokemon_to_choose)):
                self.player_team.append(pokemon_to_choose.pop(int(choice) - 1)) #.pop可以移除列表中的元素
                idx += 1
            else:
                print("请输入有效的数字来选取宝可梦!")
        print("你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)

# 电脑选择宝可梦队伍
    def computer_choose_team(self, pokemon_to_choose: list, num=3):
        sleep(1)
        print(f"电脑正在选择{num}个宝可梦")
        print()
        self.computer_team.extend(random.sample(pokemon_to_choose, num)) #.extend可以向列表中添加元素***与.append的区别***,添加一个列表时,.extend将列表中所有的元素添加到列表中,.append将列表作为整体添加到列表中
        for idx, pokemon in enumerate(self.computer_team):
            self.computer_team[idx] = copy.deepcopy(pokemon) #使用深拷贝创建新的实例,防止玩家与电脑的血量等共享
            if hasattr(self.computer_team[idx], 'name'):
                self.computer_team[idx].name += '[电脑]'
        print("电脑的宝可梦队伍:")
        self.print_pokemon_list(self.computer_team)

# 打印Pokemon列表
    def print_pokemon_list(self, pokemon_list):
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")
        print()

# 打印Pokemon的技能列表
    def print_pokemon_skills(self, pokemon):
        for skill in pokemon.skills:
            print(f"- {skill.name}: {skill.skill_description}")
# 玩家选择出战的宝可梦
    def player_choose_pokemon(self):
        print(f"你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("请输入对应数字选择出战的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if chosen_pokemon.alive is True:
                    print(f"你出战的宝可梦是{chosen_pokemon.name}")
                    self.print_pokemon_skills(chosen_pokemon)
                    print()
                    sleep(1)
                    self.player_pokemon = chosen_pokemon
                    return chosen_pokemon
                else:
                    print("你选择的宝可梦已经昏厥了,请选择别的宝可梦!")
            else:
                print("请输入有效的数字!")
            
# 电脑选择出战的宝可梦
    def computer_choose_pokemon(self):
        print(f"电脑的宝可梦队伍:")
        self.print_pokemon_list(self.computer_team)
        print("电脑正在选择出战的宝可梦......")
        print()
        sleep(1)
        alive_pokemon = [pokemon for pokemon in self.computer_team if pokemon.alive is True]
        chosen_pokemon = random.choice(alive_pokemon)
        print(f"电脑选择的宝可梦是{chosen_pokemon.name}")
        self.print_pokemon_skills(chosen_pokemon)
        print()
        sleep(1)
        self.computer_pokemon = chosen_pokemon
        return chosen_pokemon

# 玩家选择宝可梦技能
    def player_use_skills(self):
        print("选择一个技能来攻击电脑的宝可梦!")
        skills = self.player_pokemon.skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}: {skill}")
            
        choice = input("请输入对应的数字来选择一个你要使用的技能: ")
        if valid_choice(choice, len(skills)):
            player_skill = self.player_pokemon.skills[int(choice) - 1]
            self.player_pokemon.use_skill(player_skill, self.computer_pokemon)
        else:
            print("请输入有效的数字来选取技能!")

# 电脑选择宝可梦技能
    def computer_use_skills(self):
        print("电脑现在正在选择技能...")
        computer_skill = random.choice(self.computer_pokemon.skills)
        self.computer_pokemon.use_skill(computer_skill, self.player_pokemon)

# 战斗开始
    def battle_round_begin(self):
        self.player_pokemon.begin()
        self.computer_pokemon.begin()
        self.check_game_status()

# 战斗回合中
    def battle_round(self):
        print()
        print(f"{self.player_pokemon.name} VS {self.computer_pokemon.name}")
        print()
        # 判断玩家宝可梦是否被麻痹
        if any(status.name == "麻痹" for status in self.player_pokemon.statuses):
            print(f"{self.player_pokemon.name}因为麻痹而无法行动。")
        else:
            self.check_game_status()
            self.player_use_skills()
        sleep(2)
        print()
        # 判断电脑宝可梦是否被麻痹
        if any(status.name == "麻痹" for status in self.computer_pokemon.statuses):
            print(f"{self.computer_pokemon.name}因为麻痹而无法行动。")
        else:
            self.check_game_status()
            self.computer_use_skills()

# 对战循环
    def run_game(self):
        self.player_choose_team(self.all_pokemon)
        self.computer_choose_team(self.all_pokemon)
        
        print("选择一个宝可梦来战斗!")
        self.player_pokemon = self.player_choose_pokemon()
        self.computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()

# 检查游戏输赢状态
    def check_game_status(self):
        is_player_fail = all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive is False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("平局!")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("你赢了!")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("你输了!电脑战胜了你!")
            self.game_finished()
        
        if not self.player_pokemon.alive:
            self.player_choose_pokemon()
        if not self.computer_pokemon.alive:
            self.computer_choose_pokemon()

# 游戏结束
    def game_finished(self):
        sys.exit()

if __name__ == "__main__":
# 初始化宝可梦,开始运行
    print("准备好开始游戏了吗?(yes/no)")
    game = input("请输入:")
    if game.lower() == "yes":
        sleep(1)
        all_pokemon = [pokemon.PikaChu(), pokemon.Bulbasaur(), pokemon.Squirtle(), pokemon.Charmander(), pokemon.Kunkun()]
        play = Play(all_pokemon)
        play.run_game()

    else:
        print("算了吧,下次再玩啦~")