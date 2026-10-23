import typing
from .config import *
import random
from .action import *


class Pokemon:
    def __init__(
        self,
        display_name,
        max_health: int,
        health: int,
        attack: int,
        defense: int,
        miss: int,
        skill1: dict = None,
        skill2: dict = None,
        Player=False,
    ):
        self._display_name = display_name
        self.max_health = max_health
        self.health = health
        self.attack = attack
        self.defense = defense
        self.miss = miss
        self.skill1_info = skill1
        self.skill2_info = skill2
        self.effect = {}  # 效果:持续时间
        self.Player = Player

    @property
    def display_name(self):
        side = player_prefix if self.Player else enemy_prefix
        return f"{side}{self._display_name}"

    @staticmethod
    def calculate_amount(
        current: float,
        change: float,
        min_amount: int = 0,
        max_amount: int = float("inf"),
    ):
        current += change
        return max(min_amount, min(current, max_amount))

    def amount_change(
        self,
        attr_name: str,
        change: float,
        display_name: str,
        min_amount: int = 0,
        max_amount: int = float("inf"),
    ):
        value = self.calculate_amount(
            getattr(self, attr_name), change, min_amount, max_amount
        )
        value = round(value, 2)

        # if change >= 0:

        # print(amount_up_text.format(name=display_name, amount=abs(change)))
        # elif change < 0:
        # print(amount_down_text.format(name=display_name, amount=abs(change)))

        setattr(self, attr_name, value)
        if attr_name == "max_health":
            self.health = min(self.health, self.max_health)

    def health_change(self, change):
        self.amount_change("health", change, health_text, 0, self.max_health)

    def attack_change(self, change):
        self.amount_change(
            "attack",
            change,
            attack_text,
            0,
            float("inf"),
        )

    def miss_change(self, change):
        self.amount_change(
            "miss",
            change,
            miss_text,
            0,
            100,
        )

    def defense_change(self, change):
        self.amount_change(
            "defense",
            change,
            defense_text,
            0,
            float("inf"),
        )

    def max_health_change(self, change):
        self.amount_change(
            "max_health",
            change,
            max_health_text,
            0,
            float("inf"),
        )

    def ending_action(self):
        return []


class Water(Pokemon):
    element = "Water"

    def water_passive(self, change):
        if change < 0:
            resist_chance = random.randint(1, 100)
            if resist_chance <= 50:
                change *= 0.7
                print(water_passive_text.format(name=self.display_name))
                return super().health_change(change)
        elif change >= 0:
            return super().health_change(change)
        else:
            print(error_text.format(location=self.water_health_change.__name__))
        super().health_change(change)

    health_change = water_passive


class Grass(Pokemon):
    element = "Grass"

    def grass_passive(self):
        grass_heal = self.max_health * 0.1
        self.health_change(grass_heal)
        show_list = [
            grass_passive_text.format(pokemon=self.display_name, heal=grass_heal)
        ]
        return show_list

    ending_action = grass_passive


class Fire(Pokemon):
    element = "Fire"

    def __init__(
        self,
        display_name,
        max_health: int,
        health: int,
        attack: int,
        defense: int,
        miss: int,
        skill1: dict = None,
        skill2: dict = None,
        Player=False,
    ):
        super().__init__(
            display_name=display_name,
            max_health=max_health,
            health=health,
            attack=attack,
            defense=defense,
            miss=miss,
            skill1=skill1,
            skill2=skill2,
            Player=Player,
        )
        self.passive_times = 0
        self.attack_add_amount = 0.1 * self.attack



    def fire_passive(self):
        if self.effect.get("last_attack_hit", False) and self.passive_times<=3:
            self.attack_change(self.attack_add_amount)
            self.passive_times += 1
            show_list = [
                fire_passive_text.format(pokemon=self.display_name, attack=self.attack_add_amount)
            ]
            return show_list
        else:
            return []
            
    ending_action = fire_passive


class Electric(Pokemon):
    element = "Electric"

    def electric_passive(self, attacker):
        print(electric_passive_text)
        if self.Player:
            action = Act(pokemon=self, enermy_pokemon=attacker)
            return action.skill(self, attacker)
        return random.choice([self.skill1, self.skill2])(self, attacker)


class Pikachu(Electric):  # 皮卡丘
    def skill1(self, attacker, defender):
        show_list = []
        show_list.append(
            Pikachu_skill1_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )
        damage_result = damage_calcule(
            attacker=attacker,
            defender=defender,
            damage=self.attack * 1.4,
            element="Electric",
        )
        if damage_result:
            show_list.extend(damage_result)
            Pikachu_random = random.randint(1, 100)
            if (
                Pikachu_random <= 10
                and attacker.effect.get("last_attack_hit", False)
                and attacker.effect.get("last_attack_target") is defender
            ):
                defender.effect["paralysis"] = 3
                show_list.append(
                    Pikachu_paralysis_text.format(enermy_pokemon=defender.display_name)
                )
        return show_list

    def skill2(self, attacker, defender):
        show_list = []
        show_list.append(
            Pikachu_skill2_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )

        show_list.extend(
            damage_calcule(attacker=attacker, defender=defender, damage=self.attack*0.9)
        )
        Pikachu_random = random.randint(1, 100)
        if (
            Pikachu_random <= 10
            and attacker.effect.get("last_attack_hit", False)
            and attacker.effect.get("last_attack_target") is defender
        ):
            show_list.append(
                Pikachu_skill2_success_text.format(my_pokemon=attacker.display_name)
            )
            show_list.extend(
                damage_calcule(
                    attacker=attacker, defender=defender, damage=0.9 * self.attack
                )
            )
        return show_list


class Bulbasaur(Grass):  # 妙蛙种子
    def skill1(self, attacker, defender):
        show_list = []
        damage_result = damage_calcule(
            attacker=attacker,
            defender=defender,
            damage=self.attack * 1.2,
            element="Grass",
        )
        if damage_result:
            show_list.extend(damage_result)

            Bulbasaur_random = random.randint(1, 100)
            if (
                Bulbasaur_random <= 15
                and attacker.effect.get("last_attack_hit", False)
                and attacker.effect.get("last_attack_target") is defender
            ):
                defender.effect["poison"] = 3
                show_list.append(
                    Bulbasaur_skill1_success_text.format(
                        enermy_pokemon=defender.display_name
                    )
                )
        return show_list

    def skill2(self, attacker, defender):
        show_list = []
        show_list.append(
            Bulbasaur_skill2_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )
        Bulbasaur_random = random.randint(1, 100)
        if Bulbasaur_random <= 50:  # 成功
            defender.effect["absorb"] = 3
            show_list.append(
                Bulbasaur_skill2_success_text.format(
                    enermy_pokemon=defender.display_name
                )
            )
        else:
            show_list.append(
                Bulbasaur_skill2_fail_text.format(enermy_pokemon=defender.display_name)
            )
        return show_list


class Charmander(Fire):  # 小火龙
    def skill1(self, attacker, defender):
        show_list = []
        show_list.append(
            Charmander_skill1_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )

        damage_result = damage_calcule(
            attacker=attacker, defender=defender, damage=self.attack, element="Fire"
        )
        if damage_result:
            show_list.extend(damage_result)

            Chanrmander_random = random.randint(1, 100)
            if (
                Chanrmander_random <= 10
                and attacker.effect.get("last_attack_hit", False)
                and attacker.effect.get("last_attack_target") is defender
            ):
                defender.effect["burn"] = 2
                show_list.append(
                    Charmander_skill1_success_text.format(
                        enermy_pokemon=defender.display_name
                    )
                )
        return show_list

    def skill2(self, attacker, defender):
        show_list = []
        show_list.append(Charmander_skill2_text.format(pokemon=attacker.display_name))
        attacker.effect["Flame_Charge"] = 1
        return show_list


class MilkDragon(Electric):  # 奶龙
    def skill1(self, attacker, defender):
        show_list = []
        show_list.append(
            MilkDragon_skill1_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )
        MilkDragon_random = random.randint(1, 100)
        if MilkDragon_random <= 20:  # 混乱
            defender.effect["confusion"] = 3
            show_list.append(
                MilkDragon_skill1_confusion_text.format(
                    my_pokemon=attacker.display_name,
                    enermy_pokemon=defender.display_name,
                )
            )
        elif MilkDragon_random <= 40:  # 减攻击
            attack_down = defender.effect.get("MilkDragon_attack_down")
            if attack_down is None or attack_down[1] == 0:
                MilkDragon_attack_down = defender.attack * 0.2
                defender.attack_change(-MilkDragon_attack_down)
            else:
                MilkDragon_attack_down = attack_down[0]
            defender.effect["MilkDragon_attack_down"] = [
                MilkDragon_attack_down,
                3,
            ]  # 已有状态时刷新持续时间
            show_list.append(
                MilkDragon_skill1_attack_down_text.format(
                    my_pokemon=attacker.display_name,
                    enermy_pokemon=defender.display_name,
                )
            )
        elif MilkDragon_random <= 60:  # 减防
            defense_down = defender.effect.get("MilkDragon_defense_down")
            if defense_down is None or defense_down[1] == 0:
                MilkDragon_defense_down = defender.defense * 0.2
                defender.defense_change(-MilkDragon_defense_down)
            else:
                MilkDragon_defense_down = defense_down[0]
            defender.effect["MilkDragon_defense_down"] = [
                MilkDragon_defense_down,
                3,
            ]  # 已有状态时刷新持续时间
            show_list.append(
                MilkDragon_skill1_defense_down_text.format(
                    my_pokemon=attacker.display_name,
                    enermy_pokemon=defender.display_name,
                )
            )
        elif MilkDragon_random <= 80:  # 眩晕
            defender.effect["stun"] = 3
            show_list.append(
                MilkDragon_skill1_stun_text.format(
                    my_pokemon=attacker.display_name,
                    enermy_pokemon=defender.display_name,
                )
            )
        elif MilkDragon_random <= 100:  # 无事发生
            show_list.append(
                MilkDragon_skill1_nothing_text.format(
                    enermy_pokemon=defender.display_name
                )
            )
        return show_list

    def skill2(self, attacker, defender):
        show_list = []
        show_list.append(
            MilkDragon_skill2_text.format(my_pokemon=attacker.display_name)
        )
        attack_up = attacker.effect.get("MilkDragon_attack_up")
        if attack_up is None or attack_up[1] == 0:
            attack_up = attacker.attack * 0.4
            attacker.attack_change(attack_up)
        else:
            attack_up = attack_up[0]
        attacker.effect["MilkDragon_attack_up"] = [
            attack_up,
            3,
        ]  # 攻击改变值：持续时间
        return show_list


class Squirtle(Water):  # 杰尼龟
    def skill1(self, attacker, defender):
        show_list = []
        show_list.append(
            Squirtle_skill1_text.format(
                my_pokemon=attacker.display_name,
                enermy_pokemon=defender.display_name,
            )
        )
        show_list.extend(
            damage_calcule(
                attacker=attacker,
                defender=defender,
                damage=self.attack * 1.4,
                element="Water",
            )
        )
        return show_list

    def skill2(self, attacker, defender):
        show_list = []
        show_list.append(Squirtle_skill2_text.format(pokemon=attacker.display_name))
        attacker.effect["shield"] = 1
        return show_list
