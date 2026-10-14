# play.py - 游戏流程

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon

import random
from pokemon import ALL_POKEMON, Pokemon


def valid_choice(choice, max_val) -> bool:
    # 检查输入是否合法
    return choice.isdigit() and 1 <= int(choice) <= max_val


class Play:
    """游戏主控类"""

    def __init__(self) -> None:
        self.player_team= []  # 玩家队伍
        self.computer_team = []  # 电脑队伍
        self.current_player= None  # 玩家当前宝可梦
        self.current_computer= None  # 电脑当前宝可梦
        self.turn= 0  # 回合数

    def print_list(self, pokemon_list) -> None:
        # 打印宝可梦列表
        for i, p in enumerate(pokemon_list, 1):
            status = "" if p.alive else " [已昏厥]"
            print(f"{i}: {p}{status}")

    def player_choose_team(self) -> None:
        # 玩家选择3只宝可梦
        available = list(ALL_POKEMON)
        print("\n请选择3只宝可梦组成队伍（输入编号，用空格分隔）：")
        for i, c in enumerate(available, 1):
            print(f"{i}: {c.name}（{c.type}属性）")

        while True:
            choice = input("选择3只(如 1 2 3): ")
            nums = choice.split()
            if len(nums) != 3:
                print("请输入3个编号！")
                continue
            if not all(valid_choice(n, 5) for n in nums):# 检查所有编号是否有效
                print("请输入有效的编号(1-5)！")
                continue
            if len(set(nums)) != 3:
                print("不能选择重复的宝可梦！")
                continue
            for n in nums:
                c = available[int(n) - 1]
                new_p = c()#将选择的宝可梦实例化
                self.player_team.append(new_p)
                print(f"{new_p.name} 加入了队伍！")
            break

        print(f"\n你的队伍：{'、'.join(p.name for p in self.player_team)}")

    def computer_choose_team(self) -> None:
        # 电脑随机选3只
        chosen = random.sample(ALL_POKEMON, 3)
        for c in chosen:
            self.computer_team.append(c())
        print(f"电脑的队伍：{', '.join(p.name for p in self.computer_team)}")

    def player_choose_active(self) -> Pokemon:
        # 玩家选择出战宝可梦
        alive = [p for p in self.player_team if p.alive]
        print("\n选择出战的宝可梦：")
        self.print_list(alive)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        while True:
            choice = input("输入编号：")
            if valid_choice(choice, len(alive)):
                p = alive[int(choice) - 1]
                self.current_player = p
                print(f"上吧，{p.name}！")
                return p
            print("无效选择！")

    def computer_choose_active(self) -> Pokemon:
        # 电脑选择出战宝可梦
        alive = [p for p in self.computer_team if p.alive]
        p = random.choice(alive)
        self.current_computer = p
        print(f"电脑派出了 {p.name}！")
        return p

    def player_use_skill(self) -> None:
        # 玩家回合：选择技能
        p = self.current_player

        if p.charging:  # 蓄力释放
            skill = p.charged_skill
            p.charging = False
            p.charged_skill = None
            print(f"\n{p.name} 蓄力完成！")
            if getattr(skill, 'raise_enemy_evasion', False):
                self.current_computer.evasion += 0.2
            p.use_skill(skill, self.current_computer)
            if getattr(skill, 'raise_enemy_evasion', False):
                self.current_computer.evasion -= 0.2
            return

        print(f"\n为 {p.name} 选择技能：")
        for i, s in enumerate(p.skills, 1):
            tag = " [需蓄力]" if s.needs_charge else ""
            print(f"{i}: {s.name}{tag}")

        while True:
            choice = input("输入编号：")
            if valid_choice(choice, len(p.skills)):
                skill = p.skills[int(choice) - 1]
                if skill.needs_charge:  # 开始蓄力
                    p.charging = True
                    p.charged_skill = skill
                    print(f"{p.name} 开始蓄力！")
                    return
                p.use_skill(skill, self.current_computer)
                return
            print("无效选择！")

    def computer_use_skill(self) -> None:
        # 电脑回合：AI随机选技能
        p = self.current_computer

        if p.charging:  # 蓄力释放
            skill = p.charged_skill
            p.charging = False
            p.charged_skill = None
            print(f"\n{p.name} 蓄力完成！")
            if getattr(skill, 'raise_enemy_evasion', False):
                self.current_player.evasion += 0.2
            p.use_skill(skill, self.current_player)
            if getattr(skill, 'raise_enemy_evasion', False):
                self.current_player.evasion -= 0.2
            return

        skill = random.choice(p.skills)
        if skill.needs_charge:  # 开始蓄力
            p.charging = True
            p.charged_skill = skill
            print(f"{p.name} 开始蓄力！")
            return
        p.use_skill(skill, self.current_player)

    def check_status(self) -> str | None:
        # 检查胜负，返回"win"/"lose"/"tie"或None
        player_alive = [p for p in self.player_team if p.alive]
        comp_alive = [p for p in self.computer_team if p.alive]

        if not player_alive and not comp_alive:
            print("\n双方全部倒下了！平局！")
            return "tie"
        if not player_alive:
            print("\n你的宝可梦全部倒下了！你输了...")
            return "lose"
        if not comp_alive:
            print("\n电脑的宝可梦全部倒下了！你赢了！")
            return "win"

        if self.current_player and not self.current_player.alive:
            print(f"{self.current_player.name} 倒下了！")
            self.player_choose_active()
        if self.current_computer and not self.current_computer.alive:
            print(f"{self.current_computer.name} 倒下了！")
            self.computer_choose_active()

        return None

    def run(self) -> None:
        # 运行游戏
        print("=" * 45)
        print("   欢迎来到宝可梦对战竞技场！")
        print("=" * 45)

        self.player_choose_team()
        self.computer_choose_team()

        print("\n--- 选择首发宝可梦 ---")
        self.current_player = self.player_choose_active()
        self.current_computer = self.computer_choose_active()

        while True:
            self.turn += 1
            print(f"\n{'='*40}")
            print(f"       === 第{self.turn}回合 ===")
            print(f"{'='*40}")
            print(f"{self.current_player.name} HP: {self.current_player.hp}/{self.current_player.max_hp}")
            print(f"{self.current_computer.name} HP: {self.current_computer.hp}/{self.current_computer.max_hp}")

            print("\n--- 回合开始效果 ---")
            self.current_player.begin()
            self.current_computer.begin()

            if self.check_status():
                break

            print("\n--- 你的回合 ---")
            self.player_use_skill()
            if self.check_status():
                break

            print("\n--- 电脑回合 ---")
            self.computer_use_skill()
            if self.check_status():
                break


if __name__ == "__main__":
    game = Play()
    game.run()
