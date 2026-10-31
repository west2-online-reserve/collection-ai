from unittest.mock import patch

from pokemon import Bulbasaur, Charmander


def test_fire_passive_stops_at_four_stacks() -> None:
    charmander = Charmander(dodge_rate=0)
    target = Bulbasaur(hp=2000, defense=0, dodge_rate=0)

    with patch("pokemon.random.random", return_value=0.99), patch(
        "skills.random.random", return_value=0.99
    ):
        for _ in range(6):
            charmander.use_skill(charmander.skills[0], target)

    assert charmander.attack_boost_stacks == 4
    assert charmander.attack == 49


def test_flame_charge_needs_one_charge_turn() -> None:
    charmander = Charmander(dodge_rate=0)
    target = Bulbasaur(hp=1000, defense=0, dodge_rate=0)
    flame_charge = charmander.skills[1]

    charmander.use_skill(flame_charge, target)
    assert charmander.charging_skill is flame_charge
    assert target.hp == 1000

    with patch("pokemon.random.random", return_value=0.99), patch(
        "skills.random.random", return_value=0.99
    ):
        charmander.use_skill(flame_charge, target)

    assert charmander.charging_skill is None
    assert target.hp < 1000


if __name__ == "__main__":
    test_fire_passive_stops_at_four_stacks()
    test_flame_charge_needs_one_charge_turn()
    print("test_fire.py: PASS")
