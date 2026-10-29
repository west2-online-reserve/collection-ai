
#创造英雄类 
class Character():
    def __init__(self,hp:int,attack_distance:int = 0)->None:
        self.hp = hp
        self.attack_distance = attack_distance

#创造玩家大类
class Player():
    def __init__(self,
                 identify:str,
                 id:int,
                 hp:int,
                 is_computer:bool = False,
                 card_list:list=[[],[],[]],
                 attack_distance:int = 0,
                 near_dead:bool = False,
                 dead:bool = False,
                 ):
        self.identify = identify
        self.id = id
        self.hp = hp
        self.max_hp = hp
        self.is_computer = is_computer
        self.card_list = card_list
        self.attack_distance = attack_distance
        self.near_dead = near_dead
        self.dead = dead

    #定义扣血函数,并返回扣血量
    def reduce_hp(self,attack_value:int,opponent:Player,element:str)->int:
        #判定装备
        opponent.hp = max(0,opponent.hp - attack_value)
        return attack_value

    #定义回血函数，并返回回血量
    def recover_hp(self,recover_value:int)->int:
        self.hp = min(self.max_hp,self.hp + recover_value)

    
    
    