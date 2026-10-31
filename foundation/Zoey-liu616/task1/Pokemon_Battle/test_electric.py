from unittest.mock import patch

from pokemon import PikaChu


def test_electric_counter_does_not_recurse() -> None:
    attacker = PikaChu(hp=200, dodge_rate=100)
    defender = PikaChu(hp=200, dodge_rate=100)

    with (
        patch("pokemon.random.random", return_value=0.0),
        patch("pokemon.random.choice", side_effect=lambda skills: skills[0]),
    ):
        attacker.use_skill(attacker.skills[0], defender)

    # The first dodge produces exactly one counter attempt. The counter can be
    # dodged, but that dodge must not start another counter chain.
    assert attacker.hp == 200
    assert defender.hp == 200
    assert not attacker._is_countering
    assert not defender._is_countering


if __name__ == "__main__":
    test_electric_counter_does_not_recurse()
    print("test_electric.py: PASS")
