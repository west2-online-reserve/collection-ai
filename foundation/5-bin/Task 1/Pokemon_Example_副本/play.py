from __future__ import annotations
import copy
import random
import sys
from dataclasses import dataclass
from typing import NoReturn

import pokemon
from skills import Skill


# 使用dataclass简化数据结构定义，并使用frozen=True使其实例不可变
@dataclass(frozen=True)
class SkillAction:
    actor: pokemon.Pokemon
    skill: Skill
    is_player: bool


@dataclass(frozen=True)
class SwitchAction:
    actor: pokemon.Pokemon
    replacement: pokemon.Pokemon
    is_player: bool


BattleAction = SkillAction | SwitchAction


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
                chosen_pokemon = pokemon_to_choose.pop(int(choice) - 1)
                chosen_pokemon.side = "player"
                self.player_team.append(chosen_pokemon)
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
        chosen_team = random.sample(pokemon_to_choose, num)
        for member in chosen_team:
            member.side = "computer"
        self.computer_team.extend(chosen_team)
        print("Here is your opponent's team:")
        self.print_pokemon_list(self.computer_team)

    def print_pokemon_list(self, pokemon_list: list[pokemon.Pokemon]) -> None:
        # 打印Pokemon列表
        for i, p in enumerate(pokemon_list, 1):
            print(f"{i}: {p}")

    def switch_pokemon(self, is_player: bool, replacement: pokemon.Pokemon) -> bool:
        team = self.player_team if is_player else self.computer_team
        current = (
            self.current_player_pokemon if is_player else self.current_computer_pokemon
        )
        if replacement not in team or not replacement.alive or replacement is current:
            return False
        if current is not None:  # 首次选择成员时，没有旧成员离场
            current.on_switch_out()
        replacement.side = "player" if is_player else "computer"
        if is_player:
            self.current_player_pokemon = replacement
        else:
            self.current_computer_pokemon = replacement
        side = "You" if is_player else "Your opponent"
        print(f"{side} switched to {replacement}")
        return True

    def player_choose_pokemon(self) -> pokemon.Pokemon:
        # 玩家选择当前战斗的Pokemon
        print("Your Team:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("Select your pokemon to battle by number:")
            if valid_choice(choice, len(self.player_team)):
                chosen_pokemon = self.player_team[int(choice) - 1]
                if self.switch_pokemon(True, chosen_pokemon):
                    return chosen_pokemon  # 返回选择的Pokemon
                else:
                    print("Choose a living Pokemon other than the current Pokemon!")
            else:
                print("Invalid choice, please select a valid Pokemon")

    def computer_choose_pokemon(self) -> pokemon.Pokemon:
        # 电脑随机选择一个存活的Pokemon
        available_pokemon = [
            p
            for p in self.computer_team
            if p.alive and p is not self.current_computer_pokemon
        ]
        chosen_pokemon = random.choice(available_pokemon)
        self.switch_pokemon(False, chosen_pokemon)
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

    def choose_player_skill(
        self, actor: pokemon.Pokemon, allow_cancel: bool = False
    ) -> Skill | None:  # allow_cancel参数用于控制是否允许玩家取消选择技能
        print(f"Choose the skill for {actor.display_name}")
        while True:
            for i, skill in enumerate(actor.skills, 1):
                print(f"{i}: {skill} (priority {skill.priority:+d})")  # 显示技能优先级
            if allow_cancel:
                print("0: Back")
            choice = input("Select the skill you want to use by number:")
            if allow_cancel and choice == "0":
                return None
            if valid_choice(choice, len(actor.skills)):
                return actor.skills[int(choice) - 1]
            print("Invalid choice, please select a valid Skill")

    def choose_player_switch(self, actor: pokemon.Pokemon) -> pokemon.Pokemon | None:
        while True:
            print("Your Team:")
            self.print_pokemon_list(self.player_team)
            print("0: Back")
            choice = input("Select a Pokemon to switch to by number:")
            if choice == "0":
                return None
            if valid_choice(choice, len(self.player_team)):
                replacement = self.player_team[int(choice) - 1]
                if replacement.alive and replacement is not actor:
                    return replacement
            print(
                "Invalid choice, select a living teammate other than the current Pokemon"
            )

    def choose_player_action(self) -> BattleAction | None:
        actor = self.current_player_pokemon
        opponent = self.current_computer_pokemon
        if actor is None or opponent is None or not actor.alive or not opponent.alive:
            return None
        while True:
            can_switch = any(p.alive and p is not actor for p in self.player_team)
            print(f"Choose action for {actor}")
            print("1: Use a skill")
            if can_switch:
                print("2: Switch Pokemon")
            choice = input("Select your action by number:")
            if choice == "1":
                skill = self.choose_player_skill(actor, allow_cancel=True)
                if skill is not None:
                    return SkillAction(actor, skill, True)
            elif choice == "2" and can_switch:
                replacement = self.choose_player_switch(actor)
                if replacement is not None:
                    return SwitchAction(actor, replacement, True)
            else:
                print("Invalid choice, please select a valid action")

    def choose_computer_action(self) -> SkillAction | None:
        actor = self.current_computer_pokemon
        opponent = self.current_player_pokemon
        if actor is None or opponent is None or not actor.alive or not opponent.alive:
            return None
        return SkillAction(actor, random.choice(actor.skills), False)

    def order_actions(
        self, player_action: BattleAction, computer_action: BattleAction
    ) -> list[BattleAction]:  # 返回按顺序排列的行动列表，先手的行动在前
        def order_key(action: BattleAction) -> tuple[int, int, int]:
            # 返回一个用于排序的元组，优先级顺序为：是否为换人行动、技能优先级、速度
            if isinstance(action, SwitchAction):
                return (1, 0, action.actor.speed)
            return (0, action.skill.priority, action.actor.speed)

        player_key = order_key(player_action)
        computer_key = order_key(computer_action)
        # 元组越大的行动越先执行，若元组相同则随机决定先后顺序
        if player_key > computer_key:
            return [player_action, computer_action]
        if player_key < computer_key:
            return [computer_action, player_action]
        if random.choice([True, False]):
            return [player_action, computer_action]
        return [computer_action, player_action]

    def execute_action(self, action: BattleAction) -> bool:
        current = (
            self.current_player_pokemon
            if action.is_player
            else self.current_computer_pokemon
        )
        # 只允许原成员执行自己的行动，替补不能继承旧行动。
        if current is not action.actor or not action.actor.alive:
            return False
        if isinstance(action, SwitchAction):
            return self.switch_pokemon(action.is_player, action.replacement)
        opponent = (
            self.current_computer_pokemon
            if action.is_player
            else self.current_player_pokemon
        )
        if (
            opponent is None
            or not opponent.alive
            or action.skill not in action.actor.skills
        ):
            return False
        # 到实际执行时才消耗麻痹，让先手新施加的麻痹也能阻止后手。
        if action.actor.can_act():
            action.actor.use_skill(action.skill, opponent)
        return True

    def use_extra_skill(self, is_player: bool) -> None:
        actor = (
            self.current_player_pokemon if is_player else self.current_computer_pokemon
        )
        opponent = (
            self.current_computer_pokemon if is_player else self.current_player_pokemon
        )
        if actor is None or opponent is None or not actor.alive or not opponent.alive:
            return
        if not actor.can_act():
            return
        skill = (
            self.choose_player_skill(actor)
            if is_player
            else random.choice(actor.skills)
        )
        if skill is not None:
            actor.use_skill(skill, opponent)

    def player_use_skills(self) -> None:
        # 追加机会只选择技能，不开放普通行动的换人菜单。
        self.use_extra_skill(True)

    def computer_use_skills(self) -> None:
        self.use_extra_skill(False)

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
        self.turn += 1
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
            f"\n{self.current_player_pokemon.display_name} vs "
            f"{self.current_computer_pokemon.display_name}"
        )
        player_action = self.choose_player_action()
        computer_action = self.choose_computer_action()
        if player_action is None or computer_action is None:
            return
        actions = self.order_actions(player_action, computer_action)
        for action in actions:
            if not self.execute_action(action):
                continue
            self.check_game_status()
            if isinstance(action, SkillAction):
                self.process_extra_action(not action.is_player)  # 先处理防守方
                self.process_extra_action(action.is_player)

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
        pokemon1,
        pokemon2,
        pokemon3,
        pokemon4,
        pokemon5,
    ]
    play = Play(all_pokemon)
    play.run()
