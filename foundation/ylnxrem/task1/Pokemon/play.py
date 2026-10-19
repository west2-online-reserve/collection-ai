from __future__ import annotations
import copy
import random
import sys

import pokemon, skills


def valid_choice(choice: str, range: int) -> bool:
    return choice.isdigit() and 1 <= int(choice) <= range


class Play:
    def __init__(self, all_pokemon: list[type[pokemon.Pokemon]]) -> None:
        self.all_pokemon: list[type[pokemon.Pokemon]] = all_pokemon
        self.player_team: list[pokemon.Pokemon] = []
        self.computer_team: list[pokemon.Pokemon] = []
        self.current_player_pokemon: pokemon.Pokemon | None = None
        self.current_computer_pokemon: pokemon.Pokemon | None = None
        self.turn: int = 0

    def player_choose_pokemon_team(
        self, pokemon_to_choose: list[type[pokemon.Pokemon]], num: int = 3
    ) -> None:
        print(f"为你的团队选择{num}名宝可梦:")
        pokemon_to_choose = copy.copy(pokemon_to_choose)
        index = 1
        while len(self.player_team) < num:
            self.print_pokemon_list(pokemon_to_choose)
            choice = input(f"通过数字选择{index}号宝可梦: ")
            if valid_choice(choice, len(pokemon_to_choose)):
                chosen_class = pokemon_to_choose.pop(int(choice) - 1)
                self.player_team.append(chosen_class())
                index += 1
            else:
                print("请输入正确的数字")
        print("这是你的宝可梦队伍:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_pokemon_team(
        self, pokemon_to_choose: list[type[pokemon.Pokemon]], num: int = 3
    ) -> None:
        print(f"你的对手正在选择{num}只宝可梦")
        chosen_classes = random.sample(pokemon_to_choose, num)
        self.computer_team.extend([cls() for cls in chosen_classes])
        print("这是你对手的队伍:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list: list) -> None:
        for i, p in enumerate(pokemon_list, 1):
            if isinstance(p, type):
                print(f"{i}: {p.name} 属性: {p.type}")
            else:
                print(f"{i}: {p}")

    def player_choose_pokemon(self) -> pokemon.Pokemon:
        print("你的队伍:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("通过数字选择出战的宝可梦:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                print(f"你选择了 {chosen_pokemon.name}")
                self.current_player_pokemon = chosen_pokemon
                return chosen_pokemon

            else:
                print("请输入正确的数字")

    def computer_choose_pokemon(self) -> pokemon.Pokemon:
        available_pokemon = [p for p in self.computer_team if p.alive is True]
        chosen_pokemon = random.choice(available_pokemon)
        print(f"你的对手选择了{chosen_pokemon}")
        self.current_computer_pokemon = chosen_pokemon
        return chosen_pokemon

    def game_finished(self) -> None:
        sys.exit()

    def check_game_status(self) -> None:
        is_player_fail = all(p.alive is False for p in self.player_team)
        is_computer_fail = all(p.alive is False for p in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("你和对手的宝可梦都阵亡了，平局")
            self.game_finished()
        elif not is_player_fail and is_computer_fail:
            print("你的对手的宝可梦都濒死了，你赢得了这场战斗")
            self.game_finished()
        elif is_player_fail and not is_computer_fail:
            print("你的宝可梦全都濒死了。你输掉了这场战斗")
            self.game_finished()

        if self.current_player_pokemon and not self.current_player_pokemon.alive:
            self.current_player_pokemon = self.player_choose_pokemon()
        if self.current_computer_pokemon and not self.current_computer_pokemon.alive:
            self.current_computer_pokemon = self.computer_choose_pokemon()

    def switch_player_pokemon(self) -> bool:
        #实现换人
        available = [
            x for x in self.player_team if x.alive and x != self.current_player_pokemon
        ]
        if len(available) == 0:
            print("没有可以替换的宝可梦")
            return False
        print("\n可替换的宝可梦:")
        while True:
            for i, p in enumerate(available, 1):
                print(f"{i}: {p.name} (血量: {p.hp}/{p.max_hp})")
            print("0:取消换人")
            choice = input("请输入数字选择想让换上场的宝可梦：")
            if choice == "0":
                return False
            if valid_choice(choice, len(available)):
                self.current_player_pokemon = available[int(choice) - 1]
                print(f"你换上了{self.current_player_pokemon.name}")
                return True
            else:
                print("请输入正确的数字")

    def player_choose_skills(self, allow_switch=True) -> skills.Skill | None:
        if self.current_player_pokemon is None:
            return None
        if not self.current_player_pokemon.action:
            print(f"\n{self.current_player_pokemon.name} 被麻痹了，无法使用技能！")
            if allow_switch:
                choice = input("输入 1 换人，输入其他任意键跳过回合：")
                if choice == "1":
                    if self.switch_player_pokemon():
                        return "switch"
            return None  
        skills_list = self.current_player_pokemon.skills
        while True:
            for i, skill in enumerate(skills_list, 1):
                print(f"{i}: {skill.name}")
            if allow_switch:  # 只有能够换人的情况才能换人
                print(f"{len(skills_list) + 1}: 切换宝可梦")
            else:
                if not self.current_player_pokemon.action:
                    print("宝可梦无法行动")  # 在反击等无法换人的情况时被麻痹
                    break
            choice = input("通过输入数字选择你要使用的技能:")
            if allow_switch and choice == str(len(skills_list) + 1):
                if self.switch_player_pokemon():
                    return "switch"

            if valid_choice(choice, len(skills_list)):
                return self.current_player_pokemon.skills[int(choice) - 1]
            else:
                print("请输入正确的数字")

    def computer_choose_skills(self) -> skills.Skill | None:
        if self.current_computer_pokemon is None:
            return None
        if self.current_computer_pokemon.action:
            return random.choice(self.current_computer_pokemon.skills)
        return None

    def handle_evasion(
        self, defender: pokemon.Pokemon, attacker: pokemon.Pokemon
    ) -> None:
        # 处理电系宝可梦的闪避特性
        if defender.judge_evasion():
            defender.evasion = False
            if defender.chargingskill is None:
                if defender is self.current_player_pokemon:
                    skill = self.player_choose_skills(allow_switch=False)
                else:
                    skill = self.computer_choose_skills()

                if skill is not None:
                    defender.on_evade(attacker, skill)
            else:
                defender.use_skill(defender.chargingskill, attacker)

    def battle_skill(
        self, player_skill: skills.Skill | None, computer_skill: skills.Skill | None
    ) -> None:
        actions = self.judge_order(player_skill, computer_skill)
        for attacker, skill in actions:
            if not attacker.alive:
                break

            if attacker is self.current_player_pokemon:
                opponent = self.current_computer_pokemon
            else:
                opponent = self.current_player_pokemon

            if opponent is not None:
                attacker.use_skill(skill, opponent)
                self.handle_evasion(defender=opponent, attacker=attacker)

    def judge_order(
        self, player_skill: skills.Skill | None, computer_skill: skills.Skill | None
    ) -> list[tuple[pokemon.Pokemon, skills.Skill]]:
        # 根据技能优先级和宝可梦速度判断先手
        actions = []
        if player_skill is not None and self.current_player_pokemon is not None:
            actions.append((self.current_player_pokemon, player_skill))
        if computer_skill is not None and self.current_computer_pokemon is not None:
            actions.append((self.current_computer_pokemon, computer_skill))

        actions.sort(key=lambda x: (x[1].priority, x[0].speed), reverse=True)
        return actions

    def battle_round_begin(self) -> None:
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return
        self.current_player_pokemon.begin()
        self.current_computer_pokemon.begin()
        self.current_player_pokemon.action = (
            True  # 重置action属性，防止被一直麻痹无法出招
        )
        self.current_computer_pokemon.action = True
        self.current_player_pokemon.evasion = False  # 重置闪避属性防止残留
        self.current_computer_pokemon.evasion = False

        self.current_player_pokemon.apply_status_effect(self.current_computer_pokemon)
        self.check_game_status()
        self.current_computer_pokemon.apply_status_effect(self.current_player_pokemon)
        self.check_game_status()

    def battle_round(self) -> None:
        if self.current_player_pokemon is None or self.current_computer_pokemon is None:
            return

        print(
            f"\n{self.current_player_pokemon.name} vs {self.current_computer_pokemon.name}"
        )

        if not self.current_player_pokemon.charging:  # 判断是否在蓄力状态
            player_skill = self.player_choose_skills()
        else:
            player_skill = self.current_player_pokemon.chargingskill

        if not self.current_computer_pokemon.charging:
            computer_skill = self.computer_choose_skills()
        else:
            computer_skill = self.current_computer_pokemon.chargingskill
        if player_skill == "switch":
            self.current_computer_pokemon.use_skill(
                player_skill, self.current_player_pokemon
            )
            self.handle_evasion(
                self.current_player_pokemon, self.current_computer_pokemon
            )
            self.current_player_pokemon.action = None
        self.battle_skill(player_skill, computer_skill)
        self.check_game_status()

    def run(self) -> None:
        self.player_choose_pokemon_team(self.all_pokemon)
        self.computer_choose_pokemon_team(self.all_pokemon)

        print("选择你的第一只出战宝可梦")
        self.current_player_pokemon = self.player_choose_pokemon()
        self.current_computer_pokemon = self.computer_choose_pokemon()

        while True:
            self.battle_round_begin()
            self.battle_round()


if __name__ == "__main__":
    all_pokemon = [x for x in pokemon.Pokemon_Pool]
    play = Play(all_pokemon)
    play.run()
