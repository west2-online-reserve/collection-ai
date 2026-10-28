import random
import effects
from .base import Skill

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon
    from weather import Weather

class SeedBomb(Skill):
    name = "种子炸弹"
    type = "草属性"

    def __init__(self, damage: int, activation_chance: int = 15):
        super().__init__()
        self.damage = damage
        self.activation_chance = activation_chance  # 确保激活几率被正确初始化

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        # 造成伤害
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type = "physical", weather = weather)

        # 判断是否触发状态效果
        if result not in ("electric_dodge", "dodge"):
            if random.randint(1, 100) <= self.activation_chance:
                poison = next((s for s in opponent.statuses if isinstance(s, effects.PoisonEffect)),None)
                if poison:
                    poison.duration = 3
                    print(f"{self.name} 的中毒时间重置了！")
                else:
                    opponent.add_status_effect(effects.PoisonEffect())
                    print(f"{self.name} 使 {opponent.name} 中毒了!")
        return result


class ParasiticSeeds(Skill):
    name = "寄生种子"
    type = "草属性"

    def __init__(self, amount: int):
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        if random.randint(1,100) > opponent.dodge:
            # 给使用者添加治疗效果
            heal = next((s for s in user.statuses if isinstance(s, effects.ParasiticHealEffect)),None)
            if heal:
                heal.duration = 3
            else:
                user.add_status_effect(effects.ParasiticHealEffect())
            # 给对手添加中毒效果
            poison = next((s for s in opponent.statuses if isinstance(s, effects.PoisonEffect)),None)
            if poison:
                poison.duration = 3
            else:
                opponent.add_status_effect(effects.PoisonEffect())
            print("使对手中毒， 并治疗自己！")
        else:
            print(f"{opponent.name} 闪避了本次攻击！")


class MegaDrain(Skill):
    name = "终极吸取"
    type = "草属性"

    def __init__(self, damage: int):
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        damage = int(self.damage * 0.8 * mult)
        result = opponent.receive_damage(damage, damage_type="special", weather=weather)
        if result not in ("electric_dodge", "dodge"):
            heal_amt = int(damage * 0.4)
            user.heal_self(heal_amt)
            print(f"{user.name} 吸取了对方的生命力！")
        return result


class VineWhip(Skill):
    name = "藤鞭"
    type = "草属性"

    def __init__(self, damage: int):
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type="special", weather = weather)
        return result


class PoisonPowder(Skill):
    name = "催眠毒粉"
    type = "草属性"

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        print(f"使对手陷入猛毒！")
        if random.randint(1,100) > opponent.dodge:
            poison = next((s for s in opponent.statuses if isinstance(s, effects.PoisonEffect)), None)
            if poison:
                poison.duration = 4
            else:
                opponent.add_status_effect(effects.PoisonEffect())
        else:
            print(f"{opponent.name} 闪避了本次攻击！")

class RazorLeaf(Skill):
    name = "飞叶快刀"
    type = "草属性"

    def __init__(self, damage: int) -> None:
        super().__init__()
        self.damage = damage

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        mult = weather.get_weather_mult(self.type)
        result = opponent.receive_damage(int(self.damage * mult), damage_type="physical", weather=weather)
        return result


class SunnyDay(Skill):
    name = "大晴天"
    type = "草属性"

    def execute(self, user: "Pokemon", opponent: "Pokemon", weather: "Weather"):
        weather.cover_weather(weather = "Sunny", duration = 3)