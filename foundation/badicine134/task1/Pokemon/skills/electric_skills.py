import random
import effects
from .base import Skill

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
    from weather import Weather

class Thunderbolt(Skill):
    name = "十万伏特"
    type = "电属性"

    def __init__(self, damage: int, activation_chance: int = 10):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        damage = int(self.damage * mult * 1.4)
        result = opponent.receive_damage(damage, damage_type="special", weather = weather)

        if result not in ("electric_dodge", "dodge"):
            if random.randint(1, 100) <= self.activation_chance:
                opponent.add_status_effect(effects.ParalysisEffect())
                print(f"{opponent.name} 麻痹了！")
        return result

class QuickAttack(Skill):
    name = "电光一闪"
    type = "电属性"

    def __init__(self, damage: int, activation_chance: int = 10):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        damage = int(self.damage * mult)
        result = opponent.receive_damage(damage, damage_type="physical", weather = weather)
        if opponent.alive and random.randint(1, 100) <= self.activation_chance:
            print("快速攻击触发了第二次攻击！")
            self.execute(user, opponent, weather)
        return result


class ThunderPunch(Skill):
    name = "雷电拳"
    type = "电属性"

    def __init__(self, damage: int):
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type="physical", weather=weather)
        return result


class IronDefense(Skill):
    name = "铁壁"
    type = "电属性"

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        user.defense = user.base_defense + 3
        print("物防大幅增强！")


class ThunderstormCall(Skill):
    name = "雷暴呼唤"
    type = "电属性"

    def __init__(self,):
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        weather.cover_weather(weather = "Thunderstorm", duration = 3)


class Thunder(Skill):
    name = "打雷"
    type = "电属性"

    def __init__(self, damage: int, activation_chance: int = 25):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type="special", weather=weather)
        if result not in ("electric_dodge", "dodge"):
            if random.randint(1, 100) <= self.activation_chance:
                opponent.add_status_effect(effects.ParalysisEffect())
                print(f"{opponent.name} 被打雷麻痹了！")
        return result