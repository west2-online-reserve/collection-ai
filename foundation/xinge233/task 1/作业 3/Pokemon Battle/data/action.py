import typing
from .config import *
import random


def damage_calcule(attacker, defender, damage: int, element=None, missable=True):
    show_list = []
    attacker.effect["last_attack_hit"] = False#如果伤害计算没有进行到最后则为False
    if attacker.effect.get("confusion", 0) > 0:
        defender = attacker

        show_list.append(confusion_text.format(pokemon=attacker.display_name))
    attacker.effect["last_attack_target"] = defender#这里防止自己攻击自己后技能效果挂到敌方头上

    if attacker.effect.get("paralysis", 0) > 0:
        paralysis_random = random.randint(1, 100)
        if paralysis_random <= 50:
            show_list.append(
                paralysis_success_text.format(
                    pokemon=attacker.display_name
                )
            )
            return show_list
    """计算伤害"""
    miss_chance = random.randint(1, 100)
    if missable and miss_chance <= defender.miss:  # 闪避了
        defender.effect["miss"] = 1
        show_list.append(
            miss_success_text.format(
                my_pokemon=attacker.display_name, enermy_pokemon=defender.display_name
            )
        )
        if defender.element == "Electric":
            show_list.append(
                electric_passive_success_text.format(
                    my_pokemon=defender.display_name,
                    enermy_pokemon=attacker.display_name,
                )
            )
            show_list.extend(defender.electric_passive(attacker))

        return show_list
    damage -= defender.defense
    damage = max(damage, 5)
    if defender.element == effect_relationship.get(element):  # 克制伤害翻倍
        damage *= 2
        show_list.append(counter_text)
    elif defender.element == be_effect_relationship.get(element):  # 被克制伤害减半
        damage *= 0.5
        show_list.append(be_counter_text)
    if defender.effect.get("shield", 0) > 0:
        damage *= 0.5
        show_list.append(shield_text.format(pokemon=defender.display_name))
    damage = round(damage, 2)
    show_list.append(
        damage_text.format(
            my_pokemon=attacker.display_name,
            enermy_pokemon=defender.display_name,
            damage=damage,
        )
    )
    defender.health_change(-damage)
    attacker.effect["last_attack_hit"] = True
    return show_list


class Act:
    def __init__(self, pokemon=None, enermy_pokemon=None):
        self.pokemon = pokemon
        self.enermy_pokemon = enermy_pokemon

    def fight(self, attacker, defender):
        damage = attacker.attack
        show_list = []
        if attacker.effect.get("MilkDragon_attack_up", [0, 0])[1] > 0:#奶龙使攻击属性变为电
            show_list.extend(damage_calcule(
                attacker=attacker,
                defender=defender,
                damage=damage,
                element=attacker.element,
            ))
            return show_list
        show_list.extend(damage_calcule(attacker=attacker, defender=defender, damage=damage))
        return show_list

    def skill(self, attcker: str, defender: str):
        print(
            skill_options_text.format(
                skill1_name=attcker.skill1_info["name"],
                skill1_description=attcker.skill1_info["description"],
                skill2_name=attcker.skill2_info["name"],
                skill2_description=attcker.skill2_info["description"],
            )
        )
        while True:
            skill_choice = input(skill_options_input_text)
            if skill_choice == "1":
                return attcker.skill1(attcker, defender)

            elif skill_choice == "2":
                return attcker.skill2(attcker, defender)
            else:
                print(wrong_input)
                continue

    def check(self):
        print(check_options)

        while True:
            check_choice = input(check_input_text)
            if check_choice == "1":  # 查看我方出战精灵情况
                print(
                    check_pokemon_situation_text.format(
                        pokemon_name=self.pokemon.display_name,
                        hp=self.pokemon.health,
                        max_hp=self.pokemon.max_health,
                        attack=self.pokemon.attack,
                        defense=self.pokemon.defense,
                        miss=self.pokemon.miss,
                    )
                )
                break
            elif check_choice == "2":  # 查看敌方精灵情报
                print(
                    check_enermy_pokemon_situation_text.format(
                        pokemon_name=self.enermy_pokemon.display_name,
                        hp=self.enermy_pokemon.health,
                        max_hp=self.enermy_pokemon.max_health,
                        attack=self.enermy_pokemon.attack,
                        defense=self.enermy_pokemon.defense,
                        miss=self.enermy_pokemon.miss,
                    )
                )
            elif check_choice == "3":  # 查看克制关系
                print(effect_relationship_text)
                break
            else:
                print(wrong_input)

    def surrender(self):
        print(surrender_text)
        self.end()

    def end(self):

        print(end_options)

        while True:
            end_choice = input(end_input_text)

            if end_choice == "1":  # 再打一回合
                from .game_flow import Main_menu

                main_menu = Main_menu()
                main_menu.start()
            elif end_choice == "2":  # 回到主菜单
                from .game_flow import Main_menu

                main_menu = Main_menu()
                main_menu.launch()
            elif end_choice == "3":  # 退出游戏
                from .game_flow import Main_menu

                main_menu = Main_menu()
                main_menu.quit_game()
            else:
                print(wrong_input)


if __name__ == "__main__":
    act = Act()
    act.surrender()
