from unittest.mock import patch

from effects import PoisonEffect
from pokemon import Bulbasaur, Charmander, PikaChu, Squirtle, Totodile


def test_type_effectiveness() -> None:
    pikachu = PikaChu()
    bulbasaur = Bulbasaur()
    squirtle = Squirtle()
    charmander = Charmander()

    assert pikachu.type_effectiveness(squirtle) == 2.0
    assert pikachu.type_effectiveness(bulbasaur) == 0.5
    assert bulbasaur.type_effectiveness(squirtle) == 2.0
    assert squirtle.type_effectiveness(charmander) == 2.0
    assert charmander.type_effectiveness(bulbasaur) == 2.0


def test_damage_and_faint() -> None:
    charmander = Charmander()

    assert charmander.receive_damage(30) == 15
    assert charmander.hp == 65

    assert charmander.receive_damage(999) == 65
    assert not charmander.alive


def test_grass_passive() -> None:
    bulbasaur = Bulbasaur()
    bulbasaur.hp = 50

    bulbasaur.begin()

    assert bulbasaur.hp == 60


def test_fire_passive() -> None:
    charmander = Charmander()

    for _ in range(6):
        charmander.on_deal_damage()

    assert charmander.attack_stacks == 4
    assert charmander.attack == 49


def test_custom_totodile() -> None:
    totodile = Totodile()

    assert totodile.name == "小锯鳄"
    assert totodile.type == "水"
    assert totodile.hp == 90
    assert [skill.name for skill in totodile.skills] == ["泡沫", "水枪"]


def test_seed_bomb_poisons() -> None:
    attacker = Bulbasaur()
    target = Bulbasaur()

    with patch("pokemon.random.randint", return_value=100), patch(
        "skills.random.random", return_value=0.0
    ):
        attacker.use_skill(attacker.skills[0], target)

    assert target.hp == 65
    assert any(isinstance(effect, PoisonEffect) for effect in target.statuses)


def test_parasitic_seeds_drain() -> None:
    attacker = Bulbasaur()
    target = Squirtle()
    attacker.hp = 60

    with patch("pokemon.random.randint", return_value=100):
        attacker.use_skill(attacker.skills[1], target)

    target.apply_status_effect()

    assert target.hp == 72
    assert attacker.hp == 68


def test_flame_charge_charge_turn() -> None:
    charmander = Charmander()
    target = Bulbasaur()

    with patch("pokemon.random.randint", return_value=100), patch(
        "skills.random.random", return_value=0.99
    ):
        charmander.use_skill(charmander.skills[1], target)
        assert target.hp == 100

        charmander.use_skill(charmander.skills[1], target)

    assert not target.alive


def test_shield_halves_damage() -> None:
    squirtle = Squirtle()
    squirtle.use_skill(squirtle.skills[1], Charmander())

    with patch("pokemon.random.random", return_value=1.0):
        assert squirtle.receive_damage(30) == 5

    assert squirtle.hp == 75


if __name__ == "__main__":
    test_type_effectiveness()
    test_damage_and_faint()
    test_grass_passive()
    test_fire_passive()
    test_custom_totodile()
    test_seed_bomb_poisons()
    test_parasitic_seeds_drain()
    test_flame_charge_charge_turn()
    test_shield_halves_damage()
    print("test_pokemon.py: PASS")
