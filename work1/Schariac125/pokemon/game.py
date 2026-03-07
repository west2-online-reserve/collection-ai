import skills
import effects
import random
import sys
import copy
import pool
from pokemon import Pokemon


class Game:
    def __init__(self, pokemon_pool) -> None:
        self.pokemon_pool = pokemon_pool
        self.player_team = []
        self.computer_team = []
        self.current_player_pokemon = None
        self.current_computer_pokemon = None
        self.turn = 0

    def check_game_status(self):
        if not any(p.alive for p in self.player_team):
            print("游戏结束，电脑胜利")
            self.finish_game()
        elif not any(p.alive for p in self.computer_team):
            print("游戏结束，玩家胜利")
            self.finish_game()

    def choose_player_team(self):
        print("请选择三只宝可梦：")
        for i, pokemon in enumerate(self.pokemon_pool, 1):
            print(f"{i}: {pokemon}")
        while len(self.player_team) < 3:
            choice = input("请输入宝可梦的编号：")
            try:
                idx = int(choice)
            except ValueError:
                print("请输入数字编号")
                continue

            if 1 <= idx <= len(self.pokemon_pool):
                # 复制对象，避免玩家与电脑共用同一个实例
                self.player_team.append(copy.deepcopy(self.pokemon_pool[idx - 1]))
            else:
                print("无效的编号")
        print("宝可梦选择完毕\n")
        print("你的队伍：")
        for p in self.player_team:
            print(p)

    def choose_computer_team(self):
        sampled = random.sample(self.pokemon_pool, 3)
        self.computer_team = [copy.deepcopy(p) for p in sampled]
        #这里的拷贝很重要，不要去修改原对象。
        print("电脑的队伍：")
        for p in self.computer_team:
            print(p)

    def player_choose_pokemon(self):
        print("请选择一只宝可梦出战：")
        for i, pokemon in enumerate(self.player_team, 1):
            print(f"{i}: {pokemon}")
        while True:
            choice = input("请输入你要出战的宝可梦的编号")
            try:
                idx = int(choice)
            except ValueError:
                print("请输入数字编号")
                continue
            #这里是被pylance狠狠修正了，然后才改成这个，其实这个逻辑应该更好
            if 1 <= idx <= len(self.player_team):
                if not self.player_team[idx - 1].alive:
                    print("这只宝可梦已经晕倒了，请选择其他宝可梦")
                    continue
                self.current_player_pokemon = self.player_team[idx - 1]
                print(f"你选择了 {self.current_player_pokemon}")
                break
            else:
                print("无效的输入")

    def computer_choose_pokemon(self):
        available_pokemon = [p for p in self.computer_team if p.alive]
        #还是一个列表推导式，直接筛选出还活着的宝可梦，避免电脑选择已经晕倒的宝可梦
        self.current_computer_pokemon = random.choice(available_pokemon)
        print(f"电脑选择了 {self.current_computer_pokemon}")

    def finish_game(self):
        print("游戏结束")
        sys.exit(0)

    def player_choose_skill(self):
        print("请选择一个技能：")
        for i, skill in enumerate(self.current_player_pokemon.skills, 1):
            print(f"{i}: {skill.name}")
        while True:
            choice = input("请输入你要使用的技能的编号：")
            try:
                idx = int(choice)
            except ValueError:
                print("请输入数字编号")
                continue

            if 1 <= idx <= len(self.current_player_pokemon.skills):
                self.player_skill = self.current_player_pokemon.skills[idx - 1]
                print(f"你选择了 {self.player_skill.name}")
                break
            else:
                print("无效的输入")
            self.current_player_pokemon.use_skill(
                self.player_skill, self.current_computer_pokemon
            )

    def computer_choose_skill(self):
        self.computer_skill = random.choice(self.current_computer_pokemon.skills)
        print(f"电脑选择了 {self.computer_skill.name}")
        self.current_computer_pokemon.use_skill(
            self.computer_skill, self.current_player_pokemon
        )

    def start_game(self):
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            print("请先选择宝可梦")
            return
        print(
            f"游戏开始！{self.current_player_pokemon} vs {self.current_computer_pokemon}"
        )
        self.check_game_status()

    def ensure_active_pokemon(self):
        if self.current_player_pokemon is None or not self.current_player_pokemon.alive:
            if any(p.alive for p in self.player_team):
                self.player_choose_pokemon()
        if (
            self.current_computer_pokemon is None
            or not self.current_computer_pokemon.alive
        ):
            if any(p.alive for p in self.computer_team):
                self.computer_choose_pokemon()

    def battle_round(self):
        self.ensure_active_pokemon()
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            self.check_game_status()
            return
        print(f"\n回合 {self.turn+1} 开始！")
        print(f"{self.current_player_pokemon} vs {self.current_computer_pokemon}")

        # 每回合开始时结算双方状态效果与被动
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()

        # 若回合开始结算导致任一方倒下，则结束本回合
        if not self.current_computer_pokemon.alive:
            self.check_game_status()
            if any(p.alive for p in self.computer_team):
                self.computer_choose_pokemon()
            self.turn += 1
            return

        if not self.current_player_pokemon.alive:
            self.check_game_status()
            if any(p.alive for p in self.player_team):
                self.player_choose_pokemon()
            self.turn += 1
            return

        self.player_choose_skill()

        if not self.current_computer_pokemon.alive:
            self.check_game_status()
            if any(p.alive for p in self.computer_team):
                self.computer_choose_pokemon()
            self.turn += 1
            return

        self.computer_choose_skill()

        if not self.current_player_pokemon.alive:
            self.check_game_status()
            if any(p.alive for p in self.player_team):
                self.player_choose_pokemon()

        self.check_game_status()
        self.turn += 1

    def run(self):
        self.choose_player_team()
        self.choose_computer_team()
        self.player_choose_pokemon()
        self.computer_choose_pokemon()
        self.start_game()
        while True:
            self.battle_round()


if __name__ == "__main__":
    play = Game(pool.ALL_POKEMON)
    play.run()
