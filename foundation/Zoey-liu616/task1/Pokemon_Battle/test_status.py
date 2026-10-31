import io
from contextlib import redirect_stdout
from unittest.mock import patch

from effects import BurnEffect, PoisonEffect
from game import Game
from pokemon import Bulbasaur, Charmander, Chikorita, PikaChu, Squirtle


def test_status_refresh_and_expiration() -> None:
    pokemon = Charmander(hp=100, defense=99)
    pokemon.add_status_effect(BurnEffect(duration=1))
    pokemon.add_status_effect(BurnEffect(duration=2))
    pokemon.add_status_effect(BurnEffect(duration=2))

    assert len(pokemon.statuses) == 1
    assert pokemon.statuses[0].duration == 2

    pokemon.apply_status_effect()
    assert pokemon.hp == 90
    assert len(pokemon.statuses) == 1

    pokemon.apply_status_effect()
    assert pokemon.hp == 80
    assert pokemon.statuses == []


def test_status_damage_uses_actual_hp_loss() -> None:
    pokemon = Charmander(hp=5)
    pokemon.add_status_effect(BurnEffect(damage=10, duration=1))

    output = io.StringIO()
    with redirect_stdout(output):
        pokemon.apply_status_effect()

    assert pokemon.hp == 0
    assert "受到了5点烧伤伤害" in output.getvalue()
    assert pokemon.statuses == []


def test_poison_uses_ten_percent_max_hp() -> None:
    pokemon = Squirtle(hp=80)
    pokemon.add_status_effect(PoisonEffect(duration=1))

    pokemon.apply_status_effect()

    assert pokemon.hp == 72
    assert pokemon.statuses == []


def test_paralysis_rule_is_the_same_for_both_sides() -> None:
    game = Game()
    player = PikaChu()
    computer = Squirtle()
    player.paralyzed = True
    computer.paralyzed = True

    assert not game._can_act(player)
    assert not game._can_act(computer)
    assert not player.paralyzed
    assert not computer.paralyzed


def test_charge_release_keeps_computer_action_and_status_settlement() -> None:
    game = Game()
    player = Charmander(hp=200, defense=0, dodge_rate=0)
    computer = Bulbasaur(hp=500, attack=20, defense=0, dodge_rate=0)
    flame_charge = player.skills[1]
    player.charging_skill = flame_charge
    player.add_status_effect(BurnEffect(duration=2))

    with (
        patch("pokemon.random.random", return_value=0.99),
        patch("skills.random.random", return_value=0.99),
        patch("game.random.choice", side_effect=lambda skills: skills[0]),
    ):
        game._player_action(player, computer)
        hp_before_computer_action = player.hp
        game._computer_action(computer, player)
        game._end_turn(player, computer)

    assert player.charging_skill is None
    assert computer.hp < computer.max_hp
    assert player.hp < hp_before_computer_action
    assert player.statuses[0].duration == 1


def test_player_and_computer_replacement() -> None:
    game = Game()
    fainted_player = PikaChu()
    fainted_player.hp = 0
    fainted_player.alive = False
    replacement_player = Chikorita()
    fainted_computer = Charmander()
    fainted_computer.hp = 0
    fainted_computer.alive = False
    replacement_computer = Squirtle()

    game.player_team = [fainted_player, replacement_player]
    game.computer_team = [fainted_computer, replacement_computer]
    game.current_player_pokemon = fainted_player
    game.current_computer_pokemon = fainted_computer

    with patch("builtins.input", return_value="1"), patch(
        "game.random.choice", side_effect=lambda pokemon: pokemon[0]
    ):
        assert game._ensure_active_pokemon()

    assert game.current_player_pokemon is replacement_player
    assert game.current_computer_pokemon is replacement_computer


def test_battle_outcome_handles_all_four_states() -> None:
    game = Game()
    player = PikaChu()
    computer = Charmander()
    game.player_team = [player]
    game.computer_team = [computer]
    assert game._battle_outcome() is None

    computer.alive = False
    assert game._battle_outcome() == "player"
    player.alive = False
    assert game._battle_outcome() == "draw"
    computer.alive = True
    assert game._battle_outcome() == "computer"


def test_complete_battle_ends() -> None:
    game = Game()
    player = PikaChu(attack=1000, dodge_rate=0)
    game.player_team = [player, Chikorita(), Squirtle()]
    game.computer_team = [
        Bulbasaur(hp=1, defense=0, dodge_rate=0),
        Squirtle(hp=1, defense=0, dodge_rate=0),
        Charmander(hp=1, defense=0, dodge_rate=0),
    ]
    game.current_player_pokemon = player
    game.current_computer_pokemon = game.computer_team[0]

    with (
        patch("builtins.input", side_effect=["1", "1", "1"]),
        patch("game.random.choice", side_effect=lambda pokemon: pokemon[0]),
        patch("pokemon.random.random", return_value=0.99),
        patch("skills.random.random", return_value=0.99),
    ):
        game.battle_loop()

    assert any(pokemon.alive for pokemon in game.player_team)
    assert not any(pokemon.alive for pokemon in game.computer_team)
    assert game._battle_outcome() == "player"


if __name__ == "__main__":
    test_status_refresh_and_expiration()
    test_status_damage_uses_actual_hp_loss()
    test_poison_uses_ten_percent_max_hp()
    test_paralysis_rule_is_the_same_for_both_sides()
    test_charge_release_keeps_computer_action_and_status_settlement()
    test_player_and_computer_replacement()
    test_battle_outcome_handles_all_four_states()
    test_complete_battle_ends()
    print("test_status.py: PASS")
