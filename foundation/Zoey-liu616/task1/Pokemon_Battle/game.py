import random
from copy import deepcopy

from pokemon import (
    Bulbasaur,
    Charmander,
    Chikorita,
    PikaChu,
    Pokemon,
    Squirtle,
)


class Game:
    def __init__(self) -> None:
        self.pokemon_pool: list[Pokemon] = [
            PikaChu(),
            Bulbasaur(),
            Squirtle(),
            Charmander(),
            Chikorita(),
        ]
        self.player_team: list[Pokemon] = []
        self.computer_team: list[Pokemon] = []
        self.current_player_pokemon: Pokemon | None = None
        self.current_computer_pokemon: Pokemon | None = None

    def choose_player_team(self) -> None:
        print("请选择3个宝可梦组成你的队伍：")
        for index, pokemon in enumerate(self.pokemon_pool, start=1):
            print(f"{index}. {pokemon.name}({pokemon.type}属性)")

        while True:
            choices = input("请输入3个编号，用空格隔开：").split()
            if len(choices) != 3 or not all(
                choice.isdigit() for choice in choices
            ):
                print("请输入恰好3个数字！")
                continue

            indices = [int(choice) for choice in choices]
            if len(set(indices)) != 3:
                print("不能重复选择同一只宝可梦！")
                continue
            if any(
                index < 1 or index > len(self.pokemon_pool)
                for index in indices
            ):
                print("编号超出范围，请重新选择！")
                continue

            self.player_team = [
                deepcopy(self.pokemon_pool[index - 1]) for index in indices
            ]
            break

        print("你的队伍是：")
        for pokemon in self.player_team:
            print(pokemon.name)

    def choose_computer_team(self) -> None:
        self.computer_team = deepcopy(random.sample(self.pokemon_pool, 3))
        print("\n电脑已经选好了3只宝可梦！")
        print("电脑的队伍是：")
        for pokemon in self.computer_team:
            print(pokemon.name)

    def start(self) -> None:
        self.choose_player_team()
        self.choose_computer_team()

        print("\n请选择你的第一只出战宝可梦：")
        for index, pokemon in enumerate(self.player_team, start=1):
            print(f"{index}. {pokemon.name} ({pokemon.type}属性)")

        while True:
            choice = input("请输入编号：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.player_team):
                self.current_player_pokemon = self.player_team[int(choice) - 1]
                break
            print("输入无效，请重新选择！")

        self.current_computer_pokemon = random.choice(self.computer_team)
        assert self.current_player_pokemon is not None
        assert self.current_computer_pokemon is not None
        print(f"\n你选择了{self.current_player_pokemon.name}!")
        print(f"电脑选择了{self.current_computer_pokemon.name}!")
        self.battle_loop()

    def _choose_player_replacement(self) -> bool:
        available = [pokemon for pokemon in self.player_team if pokemon.alive]
        if not available:
            return False

        print("\n请选择出战宝可梦：")
        for index, pokemon in enumerate(available, start=1):
            print(f"{index}. {pokemon.name} HP: {pokemon.hp}/{pokemon.max_hp}")

        while True:
            choice = input("请输入编号：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available):
                self.current_player_pokemon = available[int(choice) - 1]
                print(f"你换上了{self.current_player_pokemon.name}!")
                return True
            print("输入无效，请重新选择！")

    def _choose_computer_replacement(self) -> bool:
        available = [pokemon for pokemon in self.computer_team if pokemon.alive]
        if not available:
            return False

        self.current_computer_pokemon = random.choice(available)
        print(f"\n电脑换上了{self.current_computer_pokemon.name}!")
        return True

    def _ensure_active_pokemon(self) -> bool:
        player = self.current_player_pokemon
        if player is None or not player.alive:
            if not self._choose_player_replacement():
                return False

        computer = self.current_computer_pokemon
        if computer is None or not computer.alive:
            if not self._choose_computer_replacement():
                return False
        return True

    def _battle_outcome(self) -> str | None:
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
            print("\n恭喜你！你赢得了宝可梦对战！")
        elif outcome == "computer":
            print("\n你的队伍全部昏厥，电脑获胜！")
        else:
            print("\n双方队伍全部昏厥，本场对战平局！")

    def _can_act(self, pokemon: Pokemon) -> bool:
        if pokemon.paralyzed:
            print(f"{pokemon.name}受到麻痹影响，无法行动！")
            pokemon.paralyzed = False
            return False
        return pokemon.alive

    def _player_action(self, player: Pokemon, computer: Pokemon) -> None:
        if not self._can_act(player):
            return

        if player.charging_skill is not None:
            print(f"{player.name}正在释放蓄力技能！")
            player.use_skill(player.charging_skill, computer)
            return

        print(f"{player.name}的技能：")
        for index, skill in enumerate(player.skills, start=1):
            print(f"{index}. {skill.name}")

        while True:
            choice = input("选择一个技能进行攻击：").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(player.skills):
                player.use_skill(player.skills[int(choice) - 1], computer)
                return
            print("输入无效，请重新选择！")

    def _computer_action(self, computer: Pokemon, player: Pokemon) -> None:
        if not self._can_act(computer):
            return

        if computer.charging_skill is not None:
            computer.use_skill(computer.charging_skill, player)
            return

        computer.use_skill(random.choice(computer.skills), player)

    def _end_turn(self, player: Pokemon, computer: Pokemon) -> None:
        for pokemon in (player, computer):
            if pokemon.alive:
                pokemon.apply_status_effect()
            if pokemon.alive:
                pokemon.begin()

    def battle_loop(self) -> None:
        while True:
            outcome = self._battle_outcome()
            if outcome is not None:
                self._announce_outcome(outcome)
                return

            if not self._ensure_active_pokemon():
                outcome = self._battle_outcome()
                self._announce_outcome(outcome or "draw")
                return

            player = self.current_player_pokemon
            computer = self.current_computer_pokemon
            assert player is not None and computer is not None

            print(f"\n你的{player.name}: HP {player.hp}/{player.max_hp}")
            print(f"电脑的{computer.name}: HP {computer.hp}/{computer.max_hp}")

            self._player_action(player, computer)

            # A counterattack can knock out the player. A fainted Pokemon must
            # not be attacked again, and a fainted computer cannot take a turn.
            if player.alive and computer.alive:
                self._computer_action(computer, player)

            self._end_turn(player, computer)

            outcome = self._battle_outcome()
            if outcome is not None:
                self._announce_outcome(outcome)
                return
            print("游戏继续")
