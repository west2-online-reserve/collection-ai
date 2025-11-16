# 定义具体的宝可梦

# 更新宝可梦表

from pokemon import *
import skills

class Bulbasaur(GlassPokemon):
    name="妙蛙种子"

    MAX_HP=100
    DODGE=0.1

    skillList=[skills.SeedBomb,skills.ParasiticSeeds]
    
    def __init__(self, attack=35, defence=10,):
        super().__init__(attack, defence)

class Pikachu(ElecPokemon):
    name="皮卡丘"

    MAX_HP=80
    DODGE=0.3
    
    skillList=[skills.Thunderbolt,skills.QuickAttack]

    def __init__(self, attack=35, defence=5):
        super().__init__(attack, defence)

   
    
POKEMON_LIST=[Bulbasaur,Pikachu]