import json
import pickle
import os


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH_pickle = os.path.join(BASE_PATH, "Pokemon_Repository.pkl")
FILE_PATH_json = os.path.join(BASE_PATH, "Pokemon_Repository.json")


# 为仓库添加所有已知的宝可梦，状态为满
def initialization():
    list_pokemon:list[Pokemon] = [PikaChu, Bulbasaur, Squirtle, Charmander, Gyarados]
    list_json:list = [x.dicted() for x in list_pokemon]

    with open(FILE_PATH_pickle, "wb") as f:
        pickle.dump(list_pokemon, f)

    with open(FILE_PATH_json, "w", encoding="utf=8") as f:
        json.dump(list_json, f, ensure_ascii=False, indent=2)

# 宝可梦类
class Pokemon(object):
    name:str
    hp:float
    max_hp:float
    attribute:str
    attack:float
    defense:float
    dod:int
    skill:list[str]
    buff:dict[str, int]
    alive:bool
    tmp_increase:float
    tmp_reduce:float
    tmp_attack:float
    tmp_defence:float
    tmp_dod:int
    # 初始化
    def __init__(self, name, max_hp, attack,
                 defense, dod, skill) -> None:
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.attack = attack
        self.defense = defense
        self.dod = dod
        self.skill = skill
        self.alive = True
        self.buff = {}
        self.tmp_increase = 1
        self.tmp_reduce = 1
        self.tmp_attack = 0
        self.tmp_defence = 0
        self.tmp_dod = 0

    # 输出宝可梦字典
    def dicted(self) -> dict:
        dict1:dict = {"name":self.name, "max_hp":self.max_hp,
                      "hp":self.hp, "attack":self.attack,
                      "defense":self.defense, "dod":self.dod,
                      "skill":self.skill, "alive":self.alive,
                      "attribute":self.attribute, "buff":self.buff
                      }
        return dict1

    # buff 层数减少
    def buff_reduce(self, round:int) -> None:
        current_round_buffs_list = [buff for buff in self.buff if BUFF_ROUND_DICT[buff] == round]
        del_key_list = []
        for x in current_round_buffs_list:
            if self.buff[x] > 1:
                self.buff[x] -= 1
            elif self.buff[x] == 1:
                del_key_list.append(x)
        for x in del_key_list:
            self.buff.pop(x)
    
    # 临时属性判断
    def tmp_attribute(self) -> None:
        self.tmp_attack = 0
        self.tmp_defence = 0
        self.tmp_increase = 1
        self.tmp_reduce = 1
        self.tmp_dod = 0
        if "麻痹" in self.buff:
            self.tmp_attack -=5
        if "火被动攻击力" in self.buff:
            self.tmp_attack += -(self.buff["火被动攻击力"] * 0.1 * self.attack )
        if "闪避减少 15%" in self.buff:
            self.tmp_dod -=15

# 各属性的宝可梦子类
class grass_Pokemon(Pokemon):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attribute = "grass"
        self.buff["草被动"] = -1

class fire_Pokemon(Pokemon):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attribute = "fire"
        self.buff["火被动"] = -1

class water_Pokemon(Pokemon):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attribute = "water"
        self.buff["水被动"] = -1

class electric_Pokemon(Pokemon):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attribute = "electric"
        self.buff["电被动"] = -1

# 宝可梦输入部分
PikaChu:electric_Pokemon = electric_Pokemon("皮卡丘", 80.0, 35.0, 5.0, 30, 
                          ["十万伏特", "电光一闪"])

Bulbasaur:grass_Pokemon = grass_Pokemon("妙蛙种子", 100.0, 35.0, 10.0, 10,
                          ["种子炸弹", "寄生种子"])

Squirtle:water_Pokemon = water_Pokemon("杰尼龟", 80.0, 25.0, 20.0, 20,
                          ["水枪", "护盾"])

Charmander:fire_Pokemon = fire_Pokemon("小火龙", 80.0, 35.0, 15.0, 10,
                          ["火花", "蓄能爆炎"])

Gyarados:water_Pokemon = water_Pokemon("暴鲤龙", 90.0, 50.0, 30.0, 20, 
                         ["破坏光线", "水炮"])


# 运行初始化函数
if __name__ == "__main__":
    initialization()


BUFF_ROUND_DICT = {
    "水被动":0, "火被动":0, "电被动":0, "火被动攻击力":0,
    "无法行动":0, "护盾":0, "爆炎蓄能完毕":0, "额外行动":0,
    "闪避减少 15%":0,
    "草被动":1, "麻痹":2, "中毒":2, "种子寄生":2, "烧伤":2}