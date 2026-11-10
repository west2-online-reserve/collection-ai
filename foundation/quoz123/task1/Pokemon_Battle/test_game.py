from unittest.mock import patch

from effects import BurnEffect
from game import Game
from pokemon import Bulbasaur, Charmander, PikaChu, Squirtle


def test_team_selection() -> None:
    game = Game()

    with patch("builtins.input", return_value="1 3 4"):
        game.choose_player_team()

    assert [pokemon.name for pokemon in game.player_team] == [
        "皮卡丘",
        "杰尼龟",
        "小火龙",
    ]


def test_end_turn() -> None:
    game = Game()
    bulbasaur = Bulbasaur()
    squirtle = Squirtle()
    bulbasaur.hp = 50
    squirtle.add_status_effect(BurnEffect())

    game.active_player = bulbasaur
    game.active_computer = squirtle
    game._end_turn()

    assert bulbasaur.hp == 60
    assert squirtle.hp == 70


def test_outcome() -> None:
    game = Game()
    player = PikaChu()
    computer = Charmander()
    game.player_team = [player]
    game.computer_team = [computer]

    assert game._outcome() is None

    computer.alive = False
    assert game._outcome() == "player"

    player.alive = False
    assert game._outcome() == "draw"


if __name__ == "__main__":
    test_team_selection()
    test_end_turn()
    test_outcome()
    print("test_game.py: PASS")
