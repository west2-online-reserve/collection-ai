import copy,random
from Pokemon import ElectricPokemon,GrassPokemon,WaterPokemon,FirePokemon,pause,pause_t,pause_battle
class Game():

    def __init__(self):
        pass
    def start(self):
        self.character = []
        #皮卡丘
        self.character.append(ElectricPokemon('PikaChu',80,35,5,30,"皮卡丘"))
        #妙蛙种子
        self.character.append(GrassPokemon('Bulbasaur',100,35,10,10,"妙蛙种子"))
        #杰尼龟
        self.character.append(WaterPokemon('Squirtle',80,25,20,20,"杰尼龟"))
        #小火龙
        self.character.append(FirePokemon('Charmander',80,35,15,10,"小火龙"))
        self.player_dizzy = []
        self.computere_dizzy = []
        self.player_paralyzed = []
        self.computere_paralyzed = []
        self.turn = 0
        self.run = 1
        self.character_choice()
        

    def character_choice(self):
        pause(pause_t)
        print("请选择 3 个宝可梦用于组成你的队伍：")
        pause(pause_t)
        for i in range(len(self.character)):
            print(f"{i+1}.  {self.character[i].chinese}({self.character[i].element_chinese})",end="  ")
            pause(pause_t/5)
        print("")
        pause(pause_t)
        self.choice = input("输入数字选择你的宝可梦：").split()
        self.player_pokemon = []
        for i in range(3):
            self.player_pokemon.append(copy.copy(self.character[int(self.choice[i]) - 1]))
        #阵营初始化
        for i in self.player_pokemon:
            i.belong = '你的'
        pause(pause_t)
        print("")
        #人机宝可梦发放
        random.shuffle(self.character)
        self.computere_pokemon = []
        for i in range(3):
            self.computere_pokemon.append(copy.copy(self.character[i]))
        #阵营初始化
        for i in self.computere_pokemon:
            i.belong = '对方的'
        self.game_looping()

    def game_looping(self)->bool:
        #判断结束函数
        def is_end(self):
            for i in self.computere_pokemon:
                if i.hp <= 0:
                    i.is_dizzy = 1
            for i in self.player_pokemon:
                if i.hp <= 0:
                    i.is_dizzy = 1
            if self.computere_pokemon[0].is_dizzy and self.computere_pokemon[1].is_dizzy and self.computere_pokemon[2].is_dizzy:
                if self.player_pokemon[0].is_dizzy and self.player_pokemon[1].is_dizzy and self.player_pokemon[2].is_dizzy:
                    return 'double'
                else:
                    return 'player'
            elif self.player_pokemon[0].is_dizzy and self.player_pokemon[1].is_dizzy and self.player_pokemon[2].is_dizzy:
                return 'computer'
            else:
                return False
        #显示函数
        def show(pokemon_list:list,dizzy_list:list,paralyzed_list:list):
            for i in range(3):
                pause(pause_t/4)
                print(f"{i+1}.{pokemon_list[i].chinese}({pokemon_list[i].element_chinese})",end="  ")
                #晕厥判定
                if pokemon_list[i].is_dizzy:
                    print("(已晕厥)")
                    if not i in dizzy_list:
                        dizzy_list.append(i)
                elif i in dizzy_list:
                    dizzy_list.remove(i)
                #麻痹判定
                elif pokemon_list[i].is_paralyzed:
                    print("(已麻痹)")
                else:
                    print(f"({pokemon_list[i].hp}/{pokemon_list[i].max_hp})")
                if pokemon_list[i].is_paralyzed:
                    pokemon_list[i].is_paralyzed -= 0.25
                    if not i in paralyzed_list:
                        paralyzed_list.append(i)
                elif i in paralyzed_list:
                    paralyzed_list.remove(i)
        #状态更新函数
        def updata(self):
            for i in self.player_pokemon:
                if i.element_id == 4:
                    i.is_act = 0 
            for i in self.computere_pokemon:
                if i.element_id == 4:
                    i.is_act = 0 

        #技能释放函数
        #玩家
        def player_skill(self):
            pause(pause_t)
            print(f"你的 {self.player_current_pokemon.chinese} 的技能：")
            pause(pause_t)
            print(f"1.{SKILL[self.player_current_pokemon.name][0]}")
            pause(pause_t)
            print(f"2.{SKILL[self.player_current_pokemon.name][1]}")
            pause(pause_t)
            self.skill_choice = int(input("选择一个技能进行攻击："))
            print("=====你的回合=====")
            pause(pause_t)
            print(f"{self.player_current_pokemon.belong} {self.player_current_pokemon.chinese} 使用了 {SKILL[self.player_current_pokemon.name][self.skill_choice - 1]}！")
            pause(pause_t)
            self.player_current_pokemon.skill_act(self.skill_choice,self.computere_current_pokemon)
            pause(pause_t)
            print("")
            pause(pause_t)
            if is_end(self):
                self.run = 0
        #人机
        def computere_skill(self):
            pause(pause_t)
            self.computere_skill_choice = random.randint(1,2)
            print(f"{self.computere_current_pokemon.belong} {self.computere_current_pokemon.chinese} 使用了 {SKILL[self.computere_current_pokemon.name][self.computere_skill_choice - 1]}！")
            self.computere_current_pokemon.skill_act(self.computere_skill_choice,self.player_current_pokemon)
            pause(pause_t)
            print("")
            pause(pause_t)
            print("")
            if is_end(self):
                self.run = 0

        #战斗循环
        while(self.run):
            #回合数
            self.turn += 1
            #更新
            updata(self)
            print(f"======{self.turn} Turn =====")
            pause(pause_t)
            #玩家角色选择显示
            print("请选择你的宝可梦：")
            pause(pause_t)
            show(self.player_pokemon,self.player_dizzy,self.player_paralyzed)
            '''
            for i in range(3):
                print(f"{i+1}.{self.player_pokemon[i].chinese}({self.player_pokemon[i].element_chinese})",end="  ")
                #玩家晕厥判定
                if self.player_pokemon[i].is_dizzy:
                    print("(已晕厥)")
                    if not i in self.player_dizzy:
                        self.player_dizzy.append(i)
                elif i in self.player_dizzy:
                    self.player_dizzy.remove(i)
                #玩家麻痹判定
                elif self.player_pokemon[i].is_paralyzed:
                    print("(已麻痹)")
                else:
                    print(f"({self.player_pokemon[i].hp}/{self.player_pokemon[i].max_hp})")
                if self.player_pokemon[i].is_paralyzed:
                    self.player_pokemon[i].is_paralyzed -= 0.25
                    if not i in self.player_paralyzed:
                        self.player_paralyzed.append(i)
                elif i in self.player_paralyzed:
                    self.player_paralyzed.remove(i)
                pause(0.1)
            '''
            pause(pause_t/4)
            print("4.查看所有角色HP")
            pause(pause_t)
            #玩家角色选择
            self.player_current_pokemon_num = -1
            self.player_current_pokemon_num = int(input("输入数字决定你的选择："))
            #选择 4 时，显示信息
            while self.player_current_pokemon_num == 4:
                pause(pause_t)
                print("======你自己======")
                show(self.player_pokemon,self.player_dizzy,self.player_paralyzed)
                pause(pause_t)
                print("======对方========")
                pause(pause_t)
                show(self.computere_pokemon,self.computere_dizzy,self.computere_paralyzed)
                self.player_current_pokemon_num = int(input("输入数字决定你的选择："))
            #完成显示信息循环后，赋值当前角色
            self.player_current_pokemon = self.player_pokemon[self.player_current_pokemon_num - 1]
            #判断所选角色是否晕厥
            while self.player_current_pokemon_num - 1 in self.player_dizzy:
                pause(pause_t)
                print(f"你的 {self.player_current_pokemon.chinese} 目前已晕厥！")
                pause(pause_t)
                self.player_current_pokemon_num = int(input("请输入数字重新选择你的宝可梦："))
                self.player_current_pokemon = self.player_pokemon[self.player_current_pokemon_num - 1]
            #判断角色是否麻痹
            while self.player_current_pokemon_num - 1 in self.player_paralyzed:
                pause(pause_t)
                print(f"你的 {self.player_current_pokemon.chinese} 目前已麻痹！")
                pause(pause_t)
                self.player_current_pokemon_num = int(input("请输入数字重新选择你的宝可梦："))
                self.player_current_pokemon = self.player_pokemon[self.player_current_pokemon_num - 1]
            pause(pause_t)
            print(f"你选择了 {self.player_current_pokemon.chinese}！")
            pause(pause_t)
            #电脑选择
            #电脑晕厥判断
            for i in range(3):
                #电脑晕厥更新
                if self.computere_pokemon[i].is_dizzy:
                    if not i in self.computere_dizzy:
                        self.computere_dizzy.append(i)
                elif i in self.computere_dizzy:
                    self.computere_dizzy.remove(i)
                #电脑麻痹更新
                if self.computere_pokemon[i].is_paralyzed:
                    if not i in self.computere_paralyzed:
                        self.computere_paralyzed.append(i)
                elif i in self.computere_paralyzed:
                    self.computere_paralyzed.remove(i)
            #电脑角色选择随机
            self.computere_current_pokemon_num = random.randint(0,2)
            self.computere_current_pokemon = self.computere_pokemon[self.computere_current_pokemon_num]
            #电脑扣除禁用
            while self.computere_current_pokemon_num in self.computere_dizzy or self.computere_current_pokemon_num in self.computere_paralyzed:
                self.computere_current_pokemon_num = random.randint(0,2)
                self.computere_current_pokemon = self.computere_pokemon[self.computere_current_pokemon_num]
            pause(pause_t)
            print(f"电脑选择了 {self.computere_current_pokemon.chinese}！")
            pause(pause_t)
            print("")

            #中毒判定
            for i in self.player_pokemon:
                if i.is_poisoning:
                    pause(pause_t)
                    print(f"{i.chinese} 受中毒影响，扣除了{int(0.1 * i.hp)}点伤害，当前HP：{i.hp - int(0.1 * i.hp)} ")
                    i.hp = i.hp - int(0.1 * i.hp)
            for i in self.computere_pokemon:
                if i.is_poisoning:
                    pause(pause_t)
                    print(f"{i.chinese} 受中毒影响，扣除了{int(0.1 * i.hp)}点伤害，当前HP：{i.hp - int(0.1 * i.hp)} ")
                    i.hp = i.hp - int(0.1 * i.hp)
                    
            pause(pause_t)
            #玩家技能
            #草被动
            for i in self.player_pokemon:
                if i.element_id == 1:
                    i.passive_skill()
            player_skill(self)
            #电被动
            if self.computere_current_pokemon.element_id == 4:
                if self.computere_current_pokemon.is_act:
                    pause(pause_t)
                    print(f"{self.computere_current_pokemon.chinese} 触发电属性被动")
                    #用于调试
                    #print("电脑电触发被动") 
                    computere_skill(self)
                    self.computere_current_pokemon.is_act = 0

            pause_battle(pause_t*7)
            '''
            print(f"你的 {self.player_current_pokemon.chinese} 的技能：")
            pause(0.2)
            print(f"1.{SKILL[self.player_current_pokemon.name][0]}")
            pause(0.1)
            print(f"2.{SKILL[self.player_current_pokemon.name][1]}")
            pause(0.5)
            self.skill_choice = int(input("选择一个技能进行攻击："))
            print("=====你的回合=====")
            pause(0.1)
            print(f"{self.player_current_pokemon.chinese} 使用了 {SKILL[self.player_current_pokemon.name][self.skill_choice - 1]}！")
            pause(0.3)
            pause(0.1)
            self.player_current_pokemon.skill_act(self.skill_choice,self.computere_current_pokemon)
            pause(0.1)
            if is_end(self):
                break
            print("")
            pause(0.2)
            '''
            print("=====对方回合=====")
            #人机技能
            #草被动
            for i in self.computere_pokemon:
                if i.element_id == 1:
                    i.passive_skill()
            #技能发动
            computere_skill(self)
            #电被动
            if self.player_current_pokemon.element_id == 4:
                if self.player_current_pokemon.is_act:
                    pause(pause_t)
                    print(f"{self.player_current_pokemon.chinese} 触发电属性被动")
                    player_skill(self)
            pause_battle(pause_t*7)
            #护盾扣除
            for i in self.computere_pokemon:
                if i.defend_rate != 0:
                    i.defend_rate = max(0,i.defend_rate - 30)
                    pause(pause_t)
                    print(f"电脑的{i.chinese} 护盾失去 30%")
                    print(f"目前护盾{i.defend_rate}")
            for i in self.player_pokemon:
                if i.defend_rate != 0:
                    i.defend_rate = max(0,i.defend_rate - 30)
                    pause(pause_t)
                    print(f"你的{i.chinese} 护盾失去 30%")
                    print(f"目前护盾{i.defend_rate}")
            '''
            pause(0.1)
            print(f"{self.computere_current_pokemon.chinese} 使用了 {SKILL[self.computere_current_pokemon.name][self.skill_choice - 1]}！")
            self.computere_current_pokemon.skill_act(random.randint(1,2),self.player_current_pokemon)
            pause(0.1)
            print("")
            pause(0.1)
            print("")
            '''

        if is_end(self) == 'player':
            pause(pause_t)
            print("恭喜你胜利")
            pause(pause_t)
            print("你的阵营如下：")
            show(self.player_pokemon,self.player_dizzy,self.player_paralyzed)
        elif is_end(self) == 'computer':
            pause(pause_t)
            print("可惜了，你的三个宝可梦倒下了")
            pause(pause_t)
            print("对方的阵营如下：")
            show(self.computere_pokemon,self.computere_dizzy,self.computere_paralyzed)
        else:
            print("平局了？？？")
            show(self.player_pokemon,self.player_dizzy,self.player_paralyzed)
            show(self.computere_pokemon,self.computere_dizzy,self.computere_paralyzed)


SKILL = {"PikaChu":['十万伏特','电光一闪'],
    "Bulbasaur":['种子炸弹','寄生种子'],
    "Squirtle":['水枪','护盾'],
    "Charmander":['火花','蓄能爆炎']}

if __name__ == "__main__":
    a = Game()
    a.start()