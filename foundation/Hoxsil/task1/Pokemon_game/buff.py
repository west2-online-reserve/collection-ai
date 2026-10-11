import random

from pokemon import Pokemon
from skill import take_damage, take_cure, buff_add, odds_produce


def glass_passive(self:Pokemon, opponent:Pokemon) -> None:
    cure = 0.1 * self.max_hp
    take_cure(self, opponent, cure, "草被动")

def Paralysis(self:Pokemon, opponent:Pokemon) -> None:
    if odds_produce(15):
        print(f"{self.name} 因麻痹无法行动了！")
        buff_add(self, "无法行动", 1)

def Poison(self:Pokemon, opponent:Pokemon) -> None:
    damage = 0.1 * self.max_hp
    take_damage(opponent, self, damage, "中毒")

def Seed_Parasitism(self:Pokemon, opponent:Pokemon) -> None:
    suck = 0.1 * self.max_hp
    take_damage(opponent, self, suck, "寄生种子")
    take_cure(opponent, self, suck, "寄生种子")

def Burn(self:Pokemon, opponent:Pokemon) -> None:
    damage = 10
    take_damage(opponent, self, damage, "烧伤")

BUFF_ROUND_DICT = {
    "水被动":0, "火被动":0, "电被动":0, "火被动攻击力":0,
    "无法行动":0, "护盾":0, "爆炎蓄能完毕":0, "额外行动":0,
    "闪避减少 15%":0,
    "草被动":1, "麻痹":2, "中毒":2, "种子寄生":2, "烧伤":2}

BUFF_FUNCTOIN_DICT = {
    "草被动":glass_passive, "麻痹":Paralysis,
    "中毒":Poison, "种子寄生":Seed_Parasitism,
    "烧伤":Burn
}