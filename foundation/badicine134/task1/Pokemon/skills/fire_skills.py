import random
import effects
from .base import Skill
from .grass_skills import SunnyDay

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
    from weather import Weather


class Ember(Skill):
    name = "火花"
    type = "火属性"

    def __init__(self, damage: int, activation_chance: int = 10):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type="physical", weather = weather)

        if result not in ("electric_dodge", "dodge"):
            if random.randint(1,100) <= self.activation_chance:
                burn = next((s for s in opponent.statuses if isinstance(s, effects.BurnEffect)),None)
                if burn:
                    burn.duration = 2
                    print(f"{self.name} 的烧伤状态重置了！")
                else:
                    opponent.add_status_effect(effects.BurnEffect())
                    print(f"{opponent.name} 烧伤了！")
        # 火属性特性
            if hasattr(user, "fire_atk_increase") and result not in ("electric_dodge", "dodge"):
                user.fire_atk_increase()
        return result
        # "fire_atk_increase" 鸭子类型

class FlameCharge(Skill):
    name = "蓄能爆炎"
    type = "火属性"

    def __init__(self, damage: int, activation_chance: int = 80):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        # 第一回合蓄力
        has_charge_effect = any(isinstance(s, effects.ChargeEffect) for s in user.statuses)
        if not has_charge_effect:
            print("蓄力一回合")
            user.add_status_effect(effects.ChargeEffect())
            return
        # 第二回合发射
        for s in user.statuses[:]:
            if isinstance(s, effects.ChargeEffect):
                user.statuses.remove(s)

        print("蓄能爆炎 蓄力完成！")
        opponent.dodge += 20
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult * 3), damage_type="special", weather = weather)
        opponent.dodge -= 20

        if result not in ("electric_dodge", "dodge"):
            if random.randint(1,100) <= self.activation_chance:
                burn = next((s for s in opponent.statuses if isinstance(s, effects.BurnEffect)),None)
                if burn:
                    burn.duration = 2
                    print(f"{self.name} 的烧伤状态重置了！")
                else:
                    opponent.add_status_effect(effects.BurnEffect())
                    print(f"{opponent.name} 被 {self.name} 烧伤了！")
            # 火属性特性
            if hasattr(user, "fire_atk_increase") and result not in ("electric_dodge", "dodge"):
                user.fire_atk_increase()
        return result


class Flamethrower(Skill):
    name = "喷射火焰"
    type = "火属性"

    def __init__(self, damage: int, activation_chance: int = 15):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * 1.3 * mult), damage_type="special", weather=weather)
        if result not in ("electric_dodge", "dodge"):
            if random.randint(1, 100) <= self.activation_chance:
                burn = next((s for s in opponent.statuses if isinstance(s, effects.BurnEffect)), None)
                if burn:
                    burn.duration = 2
                    print(f"{self.name} 的烧伤状态重置了！")
                else:
                    opponent.add_status_effect(effects.BurnEffect())
                    print(f"{opponent.name} 被 {self.name} 烧伤了！")
            if hasattr(user, "fire_atk_increase") and result not in ("electric_dodge", "dodge"):
                user.fire_atk_increase()
        return result


class WillOWisp(SunnyDay):
    name = "鬼火"
    type = "火属性"

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        super().execute(user, opponent, weather)
        opponent.add_status_effect(effects.BurnEffect(damage=5))
        print("使对手轻微烧伤")


class FireBlast(Skill):
    name = "大字爆炎"
    type = "火属性"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * 1.4 * mult), damage_type="special", weather=weather)
        if result not in ("electric_dodge", "dodge"):
            if hasattr(user, "fire_atk_increase") and result not in ("electric_dodge", "dodge"):
                user.fire_atk_increase()
        return result