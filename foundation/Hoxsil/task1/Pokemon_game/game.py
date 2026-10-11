import json
import pickle
import os
import random
import time
import copy

import skill
from pokemon import Pokemon, fire_Pokemon, water_Pokemon, grass_Pokemon, electric_Pokemon, initialization
from buff import BUFF_ROUND_DICT, BUFF_FUNCTOIN_DICT


BASE_PATH:str = os.path.dirname(os.path.abspath(__file__))
FILE_PATH_pickle:str = os.path.join(BASE_PATH, "Pokemon_Repository.pkl")
FILE_PATH_json:str = os.path.join(BASE_PATH, "Pokemon_Repository.json")
# 打开所有宝可梦的列表，并复制两份
with open(FILE_PATH_pickle, "rb") as f:
    all_pokemon_list:list = pickle.load(f)
all_player_pokemon_list:list[Pokemon] = copy.deepcopy(all_pokemon_list)
all_ai_pokemon_list:list[Pokemon] = copy.deepcopy(all_pokemon_list)


# 列表查重，重 False，不重 True
def list_repeat_judgement(list1:list) -> bool:
    for x in range(len(list1)):
        for y in range(x+1,len(list1)):
            if list1[x] == list1[y]:
                return False
    return True

# 列表元素大小，皆处于 [1,5] True，否则 False
def list_range_judgement(list1:list) -> bool:
    for x in list1:
        if x <= 0 or x >= 6:
            return False
    return True

# 选择这场战斗的宝可梦队伍
def choose_pokemon_team(all_pokemon_list:list[Pokemon]) -> list[Pokemon]:

    # 获取宝可梦名字列表
    name_list:list[str] = [all_pokemon_list[x].name for x in range(len(all_pokemon_list))]

    # 选择3个宝可梦
    while True:
        num:int = 1
        print("请选择 3 个你的要带的宝可梦")
        for x in name_list:
            print(f"{num}.{x}",end="  ")
            num += 1
        print()
        choose_list:list[int] = list(map(int, input().split()))
        time.sleep(0.5*speed)
        # 检查输入的 3 个序号的合法性
        if len(choose_list) == 3:
            if list_range_judgement(choose_list):
                if list_repeat_judgement(choose_list):
                    break
                else:
                    print("请勿选择多次选择同一个宝可梦")
            else:
                print("请选择已有序号的宝可梦")
        else:
            print("你的输入数目有误，请重新选择")
    
    pokemon_list:list = []
    print("你选择了：",end="")
    for x in choose_list:
        print(f"{name_list[x - 1]}",end=" ")
        pokemon_list.append(all_pokemon_list[x - 1])
    print()
    return pokemon_list

def choose_pokemon(pokemon_list:list[Pokemon]) -> tuple[Pokemon, int]:
    name_list:list[str] = [pokemon_list[x].name for x in range(len(pokemon_list))]
    while True:
        time.sleep(0.5*speed)
        num:int = 1
        print("请选择一个你要出战的宝可梦")
        for x in name_list:
            print(f"{num}.{x}",end="  ")
            num += 1
        print()
        choose_num:int = int(input())
        time.sleep(0.5*speed)
        if choose_num >= 1 and choose_num <= 5:
            print(f"你选择了 {name_list[choose_num - 1]}")
            return (pokemon_list[choose_num - 1], choose_num - 1)
        else:
            print("请选择已有序号的宝可梦")

def ai_choose_pokemon_team(pokemon_list:list[Pokemon]):
    pokemon_team:list[Pokemon] = random.sample(pokemon_list, 3)
    print(f"电脑出战的队伍是：")
    for x in range(1, 4):
        print(f"{x}.{pokemon_team[x-1].name}", end=" ")
    print("\n\n")
    return pokemon_team

def ai_choose_pokemon(pokemon_list:list[Pokemon]) -> tuple[Pokemon, int]:
    print("电脑选择中")

    time.sleep(1*speed)

    length:int = len(pokemon_list)
    choose_num:int = random.randint(0, length - 1)
    choosed_pokemon:Pokemon = pokemon_list[choose_num]
    print(f"电脑选择了 {choosed_pokemon.name}")
    return (choosed_pokemon, choose_num)

# 结束判断与更改
def end_judgement(player:Pokemon, ai:Pokemon) -> None:
    if player.hp <= 0:
        player.hp = 0
        player.alive = False
        print(f"你的 {player.name} 倒下了")
    if ai.hp <= 0:
        ai.hp = 0
        ai.alive = False
        print(f"你击败了对方的 {ai.name}")

# player 方选择并释放技能
def skill_choose(player:Pokemon, opponent:Pokemon) -> None:
    # 技能选择
    while True:
        print(f"你的 {player.name} 的技能：")
        num:int = 0
        for x in player.skill:
            num += 1
            print(f"{num}. {x}")
        try:
            choose_num:int = int(input("使用技能："))
        except ValueError:
            print("请输入正确的序号！")
            continue
        if 1 <= choose_num <= num:
            break
        else:
            print("请输入正确的序号！")
    # 技能实现
    skill.SKILL_FUNCTION_DICT[player.skill[choose_num - 1]](player, opponent)

# ai 选择并释放技能
def ai_skill_choose(ai:Pokemon, opponent:Pokemon) -> None:
    ai_choosed_skill:str = random.choice(ai.skill)
    skill.SKILL_FUNCTION_DICT[ai_choosed_skill](ai, opponent)

# 玩家释放技能
def player_action(player:Pokemon, opponent:Pokemon) -> None:
    if "无法行动" in player.buff:
        if player.buff["无法行动"] == 1:
            player.buff.pop("无法行动")
        else:
            player.buff["无法行动"] -= 1
        print("这回合你无法行动")
    else:
        skill_choose(player, opponent)
    
# ai 释放技能
def ai_action(ai:Pokemon, opponent:Pokemon) -> None:
    if "无法行动" in ai.buff:
        if ai.buff["无法行动"] == 1:
            ai.buff.pop("无法行动")
        else:
            ai.buff["无法行动"] -= 1
        print("这回合 ai 无法行动")
    else:
        ai_skill_choose(ai, opponent)

# 额外行动判定
def addtional_action(self:Pokemon, opponent:Pokemon) -> bool:
    if "额外行动" in self.buff:
        if self.buff["额外行动"] == 1:
            self.buff.pop("额外行动")
        else:
            self.buff["额外行动"] -= 1
        return True
    else:
        return False

# self 方buff判断
def buff_judgement(self:Pokemon, opponent:Pokemon, round:int) -> None:
    for buff in [x for x in self.buff if BUFF_ROUND_DICT[x] == round]:
        BUFF_FUNCTOIN_DICT[buff](self, opponent)

# player 的回合
def player_round(player:Pokemon, opponent:Pokemon) -> None:
    player.tmp_attribute()
    opponent.tmp_attribute()

    buff_judgement(player, opponent, 1)
    player.buff_reduce(1)

    time.sleep(1*speed)
    player_action(player, opponent)

    if addtional_action(opponent, player):
        print("电脑获得了一个额外的回合")
        ai_action(opponent, player)

    buff_judgement(opponent, player, 2)
    opponent.buff_reduce(2)

# ai 的回合
def ai_round(ai:Pokemon, opponent:Pokemon) -> None:
    ai.tmp_attribute()
    opponent.tmp_attribute()

    buff_judgement(ai, opponent, 1)
    ai.buff_reduce(1)

    time.sleep(1*speed)
    ai_action(ai, opponent)

    if addtional_action(opponent, ai):
        print("你获得了一个额外的回合！")
        player_action(opponent, ai)

    buff_judgement(opponent, ai, 2)
    opponent.buff_reduce(2)


# 测试用靶子
targe:fire_Pokemon = fire_Pokemon("靶子", 10000.0, 50.0, 30.0, 20, 
                         ["发呆"])

# ========== 主程序 ==========

# 倍率选择
SPEED_DICT:dict = {'1':0, '2':2, '3':1, '4':0.5}
while True:
    print("请选择你的游戏倍率")
    print("1. 无动画\n2. 0.5倍\n3. 1倍\n4. 2倍")
    speed_input = input()
    if speed_input in ['1', '2', '3', '4']:
        break
    else:
        print("请输入一个 1-4 的整数，以将游戏速度调整至上面的倍率")
speed:float = SPEED_DICT[speed_input]

# 初始化宝可梦状态
print("初始化宝可梦状态中...")
time.sleep(1*speed)
initialization()

# 宝可梦队伍选择
ai_pokemon_list:list[Pokemon] = ai_choose_pokemon_team(copy.deepcopy(all_pokemon_list))
player_pokemon_list:list[Pokemon] = choose_pokemon_team(all_player_pokemon_list)

# 类型声明
player_pokemon:Pokemon
ai_pokemon:Pokemon
player_choose_num:int
ai_choose_num:int

# 第一轮对决的宝可梦选择
player_pokemon, player_choose_num = choose_pokemon(player_pokemon_list)
ai_pokemon, ai_choose_num = ai_choose_pokemon(ai_pokemon_list)

# 测试用靶子赋值
# ai_pokemon = targe

# 战斗循环，其中一方宝可梦全部死亡时结束
battle_num:int = 0
while True:

    time.sleep(0.5*speed)

    battle_num += 1
    # 战斗开始打印
    print(f"battle {battle_num}：", end="")
    print(f"{player_pokemon.name} 对战 {ai_pokemon.name}！")
    round_num:int = 0
    
    # 双方循环出招，一方出招后，判断双方存活状态，一方死亡结束本轮
    while True:

        time.sleep(1*speed)

        round_num += 1

        print(f"\n\n第 {round_num} 回合开始！\n")

        print("你的回合：")

        player_round(player_pokemon, ai_pokemon)

        end_judgement(player_pokemon, ai_pokemon)

        if not (player_pokemon.alive and ai_pokemon.alive):
            break
        print()

        time.sleep(1*speed)

        print("敌方回合：")
        ai_round(ai_pokemon, player_pokemon)

        end_judgement(player_pokemon, ai_pokemon)
        if not (player_pokemon.alive and ai_pokemon.alive):
            break
    print("\n\n")
    time.sleep(1*speed)

    ai_lose:bool = False
    player_lose:bool = False
    # 战斗结束判断
    if len(ai_pokemon_list) == 1 and ai_pokemon.alive == False:
        ai_lose = True
    if len(player_pokemon_list) == 1 and player_pokemon.alive == False:
        player_lose = True
    
    if ai_lose and player_lose:
        print("你和电脑两败俱伤！")
        break
    elif player_lose:
        print("你被电脑击败了！")
        break
    elif ai_lose:
        print("你成功击败了对方！")
        break

    # 选择下一个宝可梦
    if ai_pokemon.alive == False:
        ai_pokemon_list.pop(ai_choose_num)
        ai_pokemon, ai_choose_num = ai_choose_pokemon(ai_pokemon_list)

    if player_pokemon.alive == False:
        player_pokemon_list.pop(player_choose_num)
        player_pokemon, player_choose_num = choose_pokemon(player_pokemon_list)


# 存储战后宝可梦数据
player_pokemon_list_json:list = [x.dicted() for x in all_player_pokemon_list]

with open(FILE_PATH_pickle, "wb") as f:
    pickle.dump(all_player_pokemon_list, f)

with open(FILE_PATH_json, "w", encoding="utf=8") as f:
    json.dump(player_pokemon_list_json, f, ensure_ascii=False, indent=2)