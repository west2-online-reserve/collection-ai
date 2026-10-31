from pokemon import Chikorita


def test_chikorita_initialization() -> None:
    chikorita = Chikorita()

    assert chikorita.name == "Chikorita"
    assert chikorita.hp == 90
    assert chikorita.max_hp == 90
    assert chikorita.attack == 30
    assert chikorita.defense == 15
    assert chikorita.dodge_rate == 15
    assert chikorita.type == "Grass"
    assert [skill.name for skill in chikorita.skills] == [
        "Seed Bomb",
        "Parasitic Seeds",
    ]


def test_chikorita_grass_passive() -> None:
    chikorita = Chikorita()
    chikorita.hp = 50

    chikorita.begin()

    assert chikorita.hp == 59


if __name__ == "__main__":
    test_chikorita_initialization()
    test_chikorita_grass_passive()
    print("test_chikorita.py: PASS")
