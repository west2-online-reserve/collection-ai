from __future__ import annotations

import random
from copy import deepcopy

from pokemon import (
    Bulbasaur,
    Charmander,
    PikaChu,
    Pokemon,
    Squirtle,
    Totodile,
)
from skills import Skill


class Game:
    def __init__(self) -> None:
        self.pokemon_pool: list[Pokemon] = [
            PikaChu(),
            Bulbasaur(),
            Squirtle(),
            Charmander(),
            Totodile(),
        ]
        self.player_team: list[Pokemon] = []
        self.computer_team: list[Pokemon] = []
        self.active_player: Pokemon | None = None
        self.active_computer: Pokemon | None = None
        self.turn = 1

    def start(self) -> None:
        self.choose_player_team()
        self.choose_computer_team()
        self.choose_first_pokemon()
        self.battle_loop()

    def run(self) -> None:
        self.start()

    def _print_pool(self) -> None:
        for index, pokemon in enumerate(self.pokemon_pool, start=1):
            print(f"{index}. {pokemon}")

    def choose_player_team(self) -> None:
        print("请选择 3 个宝可梦用于组成你的队伍：")
        self._print_pool()

        while True:
            raw = input("输入数字选择你的宝可梦（用空格分隔）：").strip()
            parts = raw.replace("，", " ").replace(",", " ").split()
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                print("请输入恰好 3 个数字！")
                continue

            indices = [int(part) for part in parts]
            if len(set(indices)) != 3:
                print("不能重复选择同一只宝可梦！")
                continue
            if any(index < 1 or index > len(self.pokemon_pool) for index in indices):
                print("编号超出范围，请重新选择！")
                continue

            self.player_team = [
                deepcopy(self.pokemon_pool[index - 1]) for index in indices
            ]
            break

        print("\n你的队伍：")
        for pokemon in self.player_team:
            print(f"- {pokemon}")

    def choose_computer_team(self) -> None:
        self.computer_team = deepcopy(random.sample(self.pokemon_pool, 3))
        print("\n电脑已经选好了 3 只宝可梦：")
        for pokemon in self.computer_team:
            print(f"- {pokemon}")

    def choose_first_pokemon(self) -> None:
        print("\n请选择你的首发宝可梦：")
        for index, pokemon in enumerate(self.player_team, start=1):
            print(f"{index}. {pokemon}")

        while True:
            choice = input("输入数字选择你的宝可梦：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.player_team):
                chosen = self.player_team[int(choice) - 1]
                self.active_player = chosen
                break
            print("输入无效，请重新选择！")

        computer_choice = random.choice(self.computer_team)
        self.active_computer = computer_choice
        print(f"你选择了 {chosen.name}！")
        print(f"电脑选择了 {computer_choice.name}！")

    def battle_loop(self) -> None:
        while True:
            outcome = self._outcome()
            if outcome is not None:
                self._announce_outcome(outcome)
                return

            if not self._ensure_active_pokemon():
                self._announce_outcome(self._outcome() or "draw")
                return

            player = self.active_player
            computer = self.active_computer
            assert player is not None and computer is not None

            print(f"\n========== 第 {self.turn} 回合 ==========")
            print(f"你的 {player.name}：HP {player.hp}/{player.max_hp}")
            print(f"电脑的 {computer.name}：HP {computer.hp}/{computer.max_hp}")

            self._take_turn(player, computer, is_player=True)

            if not computer.alive:
                if not self._replace_computer():
                    self._announce_outcome("player")
                    return
            elif player.alive:
                self._take_turn(computer, player, is_player=False)

            if not player.alive:
                if not self._replace_player():
                    self._announce_outcome("computer")
                    return

            self._end_turn()
            self.turn += 1

    def _take_turn(
        self,
        actor: Pokemon,
        target: Pokemon,
        *,
        is_player: bool,
    ) -> None:
        if actor.flinched:
            actor.flinched = False
            print(f"{actor.name} 畏缩了，无法行动！")
            return
        if actor.paralyzed:
            actor.paralyzed = False
            print(f"{actor.name} 因麻痹无法行动！")
            return

        charged = actor.charging_skill
        if charged is not None:
            print(f"{actor.name} 释放了蓄力中的技能！")
            actor.use_skill(charged, target)
            return

        if is_player:
            skill = self._choose_player_skill(actor)
        else:
            skill = random.choice(actor.skills)
            print(f"\n电脑的 {actor.name} 准备行动。")
        actor.use_skill(skill, target)

    def _choose_player_skill(self, pokemon: Pokemon) -> Skill:
        print(f"\n你的 {pokemon.name} 的技能：")
        for index, skill in enumerate(pokemon.skills, start=1):
            print(f"{index}. {skill.name}")

        while True:
            choice = input("选择一个技能进行攻击：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(pokemon.skills):
                return pokemon.skills[int(choice) - 1]
            print("输入无效，请重新选择！")

    def _end_turn(self) -> None:
        for pokemon in (self.active_player, self.active_computer):
            if pokemon is None or not pokemon.alive:
                continue
            pokemon.apply_status_effect()
            if pokemon.alive:
                pokemon.begin()

    def _ensure_active_pokemon(self) -> bool:
        if self.active_player is None or not self.active_player.alive:
            if not self._replace_player():
                return False
        if self.active_computer is None or not self.active_computer.alive:
            if not self._replace_computer():
                return False
        return True

    def _replace_player(self) -> bool:
        available = [pokemon for pokemon in self.player_team if pokemon.alive]
        if not available:
            return False

        print("\n请选择下一只出战的宝可梦：")
        for index, pokemon in enumerate(available, start=1):
            print(f"{index}. {pokemon.name}（HP {pokemon.hp}/{pokemon.max_hp}）")

        while True:
            choice = input("输入数字选择你的宝可梦：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available):
                self.active_player = available[int(choice) - 1]
                print(f"你换上了 {self.active_player.name}！")
                return True
            print("输入无效，请重新选择！")

    def _replace_computer(self) -> bool:
        available = [pokemon for pokemon in self.computer_team if pokemon.alive]
        if not available:
            return False
        self.active_computer = random.choice(available)
        print(f"\n电脑换上了 {self.active_computer.name}！")
        return True

    def _outcome(self) -> str | None:
        player_alive = any(pokemon.alive for pokemon in self.player_team)
        computer_alive = any(pokemon.alive for pokemon in self.computer_team)

        if player_alive and computer_alive:
            return None
        if player_alive:
            return "player"
        if computer_alive:
            return "computer"
        return "draw"

    def _announce_outcome(self, outcome: str) -> None:
        if outcome == "player":
            print("\n恭喜你，你赢得了这场宝可梦对战！")
        elif outcome == "computer":
            print("\n你的宝可梦全部昏厥，电脑获胜！")
        else:
            print("\n双方宝可梦全部昏厥，本场对战平局！")
