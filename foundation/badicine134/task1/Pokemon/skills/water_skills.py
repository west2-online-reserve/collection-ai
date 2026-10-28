import effects
from .base import Skill

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
    from weather import Weather


class AquaJet(Skill):
    name = "水枪"
    type = "水属性"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * 1.3 * mult), damage_type="physical", weather = weather)
        return result


class Shield(Skill):
    name = "护盾"
    type = "水属性"

    def __init__(self,) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        print(f"下一回合受到的伤害减少 50%")
        user.add_status_effect(effects.ShieldEffect())


class Surf(Skill):
    name = "冲浪"
    type = "水属性"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * 1.2 * mult), damage_type="special", weather=weather)
        return result


class Amnesia(Skill):
    name = "健忘"
    type = "水属性"

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        user.sp_defense = user.base_sp_defense + 3
        print(f"特防大幅提升！")


class RainDance(Skill):
    name = "祈雨"
    type = "水属性"

    def __init__(self,) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        weather.cover_weather(weather = "Rain", duration = 3)


class HydroPump(Skill):
    name = "水炮"
    type = "水属性"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * 1.1 * mult), damage_type="special", weather=weather)
        return result