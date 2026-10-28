from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
    from weather import Weather

class Skill:
    name: str
    type: str

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"
