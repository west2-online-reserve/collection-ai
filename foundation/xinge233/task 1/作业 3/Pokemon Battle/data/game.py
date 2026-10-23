from .config import *
from pathlib import Path
import random
import json
from .role import *
from copy import deepcopy
from .action import Act
import sys
import time

BASE = Path(__file__).resolve().parent  # 绝对路径游戏主目录
# 提前实例化一波对象，到时候选到直接深拷贝
pokemon_classes = {
    "Pikachu": Pikachu,
    "Bulbasaur": Bulbasaur,
    "Squirtle": Squirtle,
    "Charmander": Charmander,
    "MilkDragon": MilkDragon,
}

#
#def reset() -> None:  # 清除数据
#    pass
#毙掉了

def confirm(action: str):  # 是否确定
    print(confirm_text.format(act=action))
    if sure():
        return True
    else:
        return False


def sure() -> bool:  # 是否选项
    while True:
        for i in sure_text:
            print(i)
        sure_answer = input(sure_input)
        if sure_answer == "1":
            return True
        elif sure_answer == "2":
            return False
        else:
            print(wrong_input)


#def save() -> bool:
#   try:
#       save_path = Path(BASE / "saves" / "save.json")
#        save_path.parent.mkdir(parents=True, exist_ok=True)
#        save_data = "这里先占个位，还没写"
#        with open(save_path, "w", encoding="utf-8") as s:
#            json.dump(save_data, s, ensure_ascii=False, indent=4)
#        print(save_success.format(path=save_path))
#        return True
#    except:
#        print(save_error)
#        return False
#毙掉了

class Game:

    def pick(self):
        self.winner_situation = [0, 0]  # [玩家胜利次数，敌方胜利次数]
        # 先实例化一波所有宝可梦，到时候直接deepcopy
        self.pokemon_dict = {}  # 存储所有宝可梦对象的字典
        for name, cls in pokemon_classes.items():
            self.pokemon_dict[name] = cls(
                display_name=pokemon_text_dict[name]["name"],
                max_health=pokemon_text_dict[name]["max_hp"],
                health=pokemon_text_dict[name]["hp"],
                attack=pokemon_text_dict[name]["attack"],
                defense=pokemon_text_dict[name]["defense"],
                miss=pokemon_text_dict[name]["miss"],
                skill1=pokemon_text_dict[name]["skill_1"],
                skill2=pokemon_text_dict[name]["skill_2"],
            )
        while True:
            print(organize_team_text)
            self.team = []
            for index, name in enumerate(pokemon_text_dict.keys(), start=1):
                print(
                    name_element_info.format(
                        index=index,
                        name=pokemon_text_dict[name]["name"],
                        element=pokemon_text_dict[name]["type"],
                    )
                )
                print(
                    element_passives_text.format(
                        element=pokemon_text_dict[name]["type"],
                        passive=element_passives[pokemon_text_dict[name]["type"]],
                    )
                )
                print(
                    state_info.format(
                        maxhp=pokemon_text_dict[name]["max_hp"],
                        attack=pokemon_text_dict[name]["attack"],
                        denfense=pokemon_text_dict[name]["defense"],
                        miss=pokemon_text_dict[name]["miss"],
                    )
                )
                print(
                    skill_info.format(
                        name1=pokemon_text_dict[name]["skill_1"]["name"],
                        skill1=pokemon_text_dict[name]["skill_1"]["description"],
                        name2=pokemon_text_dict[name]["skill_2"]["name"],
                        skill2=pokemon_text_dict[name]["skill_2"]["description"],
                    )
                )
            while True:
                try:
                    role1, role2, role3 = map(str, input(organize_input_text).split())
                    if role1 == role2 or role1 == role3 or role2 == role3:
                        print(wrong_input)#防止重复选择
                        continue
                    else:
                        self.team.append(num_role_relationship[role1])
                        self.team.append(num_role_relationship[role2])
                        self.team.append(num_role_relationship[role3])
                        break

                except (ValueError, KeyError):
                    print(wrong_input)
                    continue

            if len(self.team) != 3:
                print(wrong_team_text)
                continue
            elif confirm(team_comfirm_text):
                self.enermy_team = random.sample(
                    list(pokemon_classes.keys()), 3
                )  # 随机选择敌方宝可梦
                print(
                    enermy_team_text.format(
                        enermy1=f"{enemy_prefix}{pokemon_text_dict[self.enermy_team[0]]['name']}",
                        enermy2=f"{enemy_prefix}{pokemon_text_dict[self.enermy_team[1]]['name']}",
                        enermy3=f"{enemy_prefix}{pokemon_text_dict[self.enermy_team[2]]['name']}",
                    )
                )
                self.battle_ready()
                break

    def battle_ready(self):
        # 胜负判断
        if self.winner_situation[1] == 2:
            print(surrender_text)
            battle_ready_act=Act(pokemon=self.pokemon, enermy_pokemon=self.enermy_pokemon)
            battle_ready_act.end()
        if self.winner_situation[0] == 2:
            for line in Win:
                print(line)
                time.sleep(0.2)
            battle_ready_act=Act(pokemon=self.pokemon, enermy_pokemon=self.enermy_pokemon)
            battle_ready_act.end()
        # 比分展示
        print(
            winner_situation_text.format(
                score1=self.winner_situation[0], socore2=self.winner_situation[1]
            )
        )
        enermy_team_situation= [f"{enemy_prefix}{pokemon_text_dict[pokemon]['name']}" for pokemon in self.enermy_team]
        print(enermy_team_situation_text + "、".join(enermy_team_situation))
        team_choices = "\n".join(
            f"{index}.{player_prefix}{pokemon_text_dict[pokemon]['name']}"
            for index, pokemon in enumerate(self.team, start=1)
        )
        
        print(team_roles.format(team_choices=team_choices))
        enermy_pokemon = random.choice(self.enermy_team)
        # print(enermy_pokemon)
        print(
            battle_ready_text.format(
                enermy_pokemon=f"{enemy_prefix}{pokemon_text_dict[enermy_pokemon]['name']}"
            )
        )
        while True:
            try:
                fight_pokemon = self.team[
                    int(input(fight_input_text.format(team_size=len(self.team)))) - 1
                ]
                break
            except (ValueError, IndexError):
                print(wrong_input)
                continue
        self.pokemon = deepcopy(
            self.pokemon_dict[fight_pokemon]
        )  # 深拷贝选中的宝可梦对象
        self.pokemon.Player = True
        self.enermy_pokemon = deepcopy(self.pokemon_dict[enermy_pokemon])
        print(battle_start_text)
        # print(self.pokemon)
        # print(self.enermy_pokemon)
        self.battle()
        # 接下来这里要实例化宝可梦

    def battleable(self, pokemon, enermy_pokemon):  # 检测是否具有异常状态
        disability = 0
        if pokemon.health <= 0:
            if pokemon is self.pokemon:
                self.winner_situation[1] += 1
            else:
                self.winner_situation[0] += 1
            for i in self.battle_display:  # 展示战斗过程
                print(i)
                time.sleep(0.5)
            self.team.remove(self.pokemon.__class__.__name__)
            self.enermy_team.remove(self.enermy_pokemon.__class__.__name__)
            print(battle_end_text.format(pokemon=pokemon.display_name))
            self.battle_ready()
            return False

        pokemon.effect["miss"] = 0  # 每回合重置闪避状态
        if pokemon.effect.get("Flame_Charge", -1) == 1:
            self.battle_display.append(Charging_text.format(pokemon=pokemon.display_name))
            disability += 1
        if "Flame_Charge" in pokemon.effect:
            pokemon.effect["Flame_Charge"] -= 1  # 蓄能爆炎未在蓄力状态时为负数
        if pokemon.effect.get("confusion", 0) > 0:  # 混乱检测
            pokemon.effect["confusion"] -= 1

        if pokemon.effect.get("stun", 0) > 0:  # 眩晕检测
            pokemon.effect["stun"] -= 1
            self.battle_display.append(stun_text.format(pokemon=pokemon.display_name))
            disability += 1
        if pokemon.effect.get("shield", 0) > 0:  # 护盾检测
            pokemon.effect["shield"] -= 1
            self.battle_display.append(shield_disappear_text.format(pokemon=pokemon.display_name))
        if pokemon.effect.get("MilkDragon_attack_down", [0, 0])[1] > 0:  # 伤害降低检测
            pokemon.effect["MilkDragon_attack_down"][1] -= 1
            if pokemon.effect.get("MilkDragon_attack_down")[1] == 0:
                self.battle_display.append(attack_down_end_text.format(pokemon=pokemon.display_name))
                pokemon.attack_change(pokemon.effect.get("MilkDragon_attack_down")[0])
        if pokemon.effect.get("MilkDragon_defense_down", [0, 0])[1] > 0:  # 防御降低检测
            pokemon.effect["MilkDragon_defense_down"][1] -= 1
            if pokemon.effect.get("MilkDragon_defense_down")[1] == 0:
                self.battle_display.append(defense_down_end_text.format(pokemon=pokemon.display_name))
                pokemon.defense_change(pokemon.effect.get("MilkDragon_defense_down")[0])
        if pokemon.effect.get("MilkDragon_attack_up", [0, 0])[1] > 0:  # 伤害提升检测
            pokemon.effect["MilkDragon_attack_up"][1] -= 1
            if pokemon.effect.get("MilkDragon_attack_up")[1] == 0:
                self.battle_display.append(attack_up_end_text.format(pokemon=pokemon.display_name))
                pokemon.attack_change(-pokemon.effect.get("MilkDragon_attack_up")[0])

        if pokemon.effect.get("Flame_Charge", -1) == 0:
            self.battle_display.append(Flame_Charge_text.format(pokemon=pokemon.display_name,enermy_pokemon=enermy_pokemon.display_name))
            Flame_Charge_random = random.randint(1, 100)
            if  not Flame_Charge_random <= enermy_pokemon.miss + 20: #闪避时为真,not后就是闪避为假，没有闪避才执行下面的内容
                for message in damage_calcule(
                        attacker=pokemon,
                        defender=enermy_pokemon,
                        damage=pokemon.attack * 3,
                        element="Fire",
                        missable=False,
                    ):
                    self.battle_display.append(message)
                Flame_Charge_random = random.randint(1, 100)
                if Flame_Charge_random <= 80 and pokemon.effect.get("last_attack_hit",False) == True and pokemon.effect.get("last_attack_target") is enermy_pokemon:  # 如果打到人了,80%概率附加烧伤状态,如果混乱了
                    enermy_pokemon.effect["burn"] = 3
                    self.battle_display.append(
                        Flame_Charge_burn_text.format(
                            attacker=pokemon.display_name,
                            defender=enermy_pokemon.display_name,
                        )
                    )

            else:
                self.battle_display.append(
                    miss_success_text.format(
                        my_pokemon=enermy_pokemon.display_name,
                        enermy_pokemon=pokemon.display_name,
                    )
                )
        if pokemon.effect.get("burn", 0) > 0:  # 烧伤检测
            self.battle_display.append(burn_text.format(pokemon=pokemon.display_name))
            pokemon.effect["burn"] -= 1
            pokemon.health_change(-10)
            self.battle_display.append(burn_remain_round_text.format(round=pokemon.effect["burn"]))
        if pokemon.effect.get("poison", 0) > 0:  # 中毒检测
            self.battle_display.append(poison_text.format(pokemon=pokemon.display_name))
            pokemon.effect["poison"] -= 1
            pokemon.health_change(-pokemon.max_health * 0.1)
            self.battle_display.append(poison_remain_round_text.format(round=pokemon.effect["poison"]))
        if pokemon.effect.get("absorb", 0) > 0:  # 寄生检测
            self.battle_display.append(
                absorb_text.format(
                    pokemon=pokemon.display_name,
                    enermy_pokemon=enermy_pokemon.display_name,
                )
            )
            pokemon.effect["absorb"] -= 1
            pokemon.health_change(-pokemon.max_health * 0.1)
            enermy_pokemon.health_change(pokemon.max_health * 0.1)
        if pokemon.effect.get("paralysis", 0) > 0:  # 麻痹检测
            self.battle_display.append(paralysis_text.format(pokemon=pokemon.display_name))
            pokemon.effect["paralysis"] -= 1
        if pokemon.health <= 0:#最后再检查一次，避免因为异常状态导致宝可梦死亡后还继续战斗
            if pokemon is self.pokemon:
                self.winner_situation[1] += 1
            else:
                self.winner_situation[0] += 1
            for i in self.battle_display:  # 展示战斗过程
                print(i)
                time.sleep(0.5)
            self.team.remove(self.pokemon.__class__.__name__)
            self.enermy_team.remove(self.enermy_pokemon.__class__.__name__)
            print(battle_end_text.format(pokemon=pokemon.display_name))
            self.battle_ready()
            return False

        if disability > 0:
            return False
        else:
            return True

    def battle(self):
        # 初始化
        from .enermy import Enermy_action

        round = 1
        while True:
            # 进入战斗循环
            self.battle_display = [round_count_text.format(round=round)]  # 每回合的战斗信息列表
            if self.battleable(
                self.pokemon, self.enermy_pokemon
            ):  # 如果我方宝可梦还能战斗，进入战斗选择
                action = Act(
                    pokemon=self.pokemon,
                    enermy_pokemon=self.enermy_pokemon,
                )#调用前实例化，防止重新开始后旧的数据残留
                while True:
                    # 展示战斗信息
                    print(round_info.format(round=round))
                    print(
                        pokemon_situation_text.format(
                            pokemon_name=self.pokemon.display_name,
                            hp=self.pokemon.health,
                            max_hp=self.pokemon.max_health,
                        )
                    )
                    print(
                        enermy_pokemon_situation_text.format(
                            pokemon_name=self.enermy_pokemon.display_name,
                            hp=self.enermy_pokemon.health,
                            max_hp=self.enermy_pokemon.max_health,
                        )
                    )
                    print(act_tip_text.format(round=round))
                    battle_choice = input(battle_input_text)
                    # 从这里开始战斗选择
                    if battle_choice == "1":  # 普通攻击
                        self.battle_display.append(
                            act_text.format(
                                my_pokemon=self.pokemon.display_name,
                                enermy_pokemon=self.enermy_pokemon.display_name,
                                action_name=act_display_name.get("fight"),
                            )
                        )
                        self.battle_display.extend(
                            action.fight(self.pokemon, self.enermy_pokemon)
                        )
                        break
                    elif battle_choice == "2":  # 技能
                        self.battle_display.extend(
                            action.skill(self.pokemon, self.enermy_pokemon)
                        )
                        break
                    elif battle_choice == "3":  # 调查
                        action.check()
                    elif battle_choice == "4":  # 投降
                        action.surrender()
                    else:
                        print(wrong_input)
                        continue
                self.battle_display.extend(self.pokemon.ending_action())  # 执行我方宝可梦的回合结束动作
                
            if self.battleable(
                self.enermy_pokemon, self.pokemon
            ):  # 检测敌人是否可以行动
                enermy_action = Enermy_action(
                    self.enermy_pokemon,
                    self.pokemon,
                )
                self.battle_display.extend(enermy_action.enermy_action_start())
                self.battle_display.extend(self.enermy_pokemon.ending_action())  # 执行敌方宝可梦的回合结束动作
                
            # 这里先设计一下敌方战斗逻辑
            # 敌方战斗逻辑设计前要先设计完精灵的技能实现

            for i in self.battle_display:  # 展示战斗过程
                print(i)
                time.sleep(0.5)
            time.sleep(0.5)#不知道为什么，这里睡一会后会变得流畅一点
            self.enermy_pokemon.effect["last_attack_hit"] = False  # 重置击中判定
            self.enermy_pokemon.effect["last_attack_target"] = None  # 重置上次攻击目标
            self.pokemon.effect["last_attack_hit"] = False  # 重置击中判定
            self.pokemon.effect["last_attack_target"] = None  # 重置上次攻击目标
            round += 1


if __name__ == "__main__":

    # confirm("test")  # 同时测试sure
    # save()
    game = Game()
    test_choice = input("请选择测试功能:\n1.选宝可梦\n2.战斗\n请输入1/2:")
    if test_choice == "1":
        game.pick()
    elif test_choice == "2":
        game.pokemon_dict = {}  # 存储所有宝可梦对象的字典
        for name, cls in pokemon_classes.items():
            game.pokemon_dict[name] = cls(
                display_name=pokemon_text_dict[name]["name"],
                max_health=pokemon_text_dict[name]["max_hp"],
                health=pokemon_text_dict[name]["hp"],
                attack=pokemon_text_dict[name]["attack"],
                defense=pokemon_text_dict[name]["defense"],
                miss=pokemon_text_dict[name]["miss"],
            )
        print(game.pokemon_dict)

        game.pokemon = deepcopy(game.pokemon_dict["Pikachu"])
        game.enermy_pokemon = deepcopy(game.pokemon_dict["Bulbasaur"])
        game.battle()
