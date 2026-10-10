from some_function import lucky_bar,pause,pause_t,pause_battle

class Pokemon():
    def __init__(self,name:str,hp:int,attack_value:int,defend_value:int,dodge_rate:int,chinese:str):
        self.max_hp = hp
        self.hp = hp
        self.attack_value = attack_value
        self.defend_value = defend_value
        self.dodge_rate = dodge_rate
        self.name = name
        self.chinese = chinese
        self.is_dizzy = 0
        #烧伤
        self.is_fired = 0
        #火蓄力
        self.flame_count = 0
        #护盾
        self.defend_rate = 0
        #中毒
        self.is_poisoning = 0
        #麻痹
        self.is_paralyzed = 0
        #寄生种子
        self.is_seed = 0
        #所属阵营
        self.belong = None
            
    #攻击伤害计算
    def attack(self,op_character:Pokemon,base_attack_value:int)->int:
        #躲闪计算
        is_dodge = 0
        #random.randint(1,100) <= op_character.dodge_rate:
        if lucky_bar(op_character.dodge_rate/100,'能否闪躲'):
            pause(pause_t)
            print(f"{op_character.belong} {op_character.chinese} 成功闪避！")
            is_dodge = 1

        #元素被克
        #这里 -1 代表被对方克制，1 代表克制对方
        is_counter = 0
        if self.element_id == 1:
            if op_character.element_id == 3:
                is_counter = 1
            elif op_character.element_id == 2:
                is_counter = -1
        elif self.element_id == 2:
            if op_character.element_id == 1:
                is_counter = 1
            elif op_character.element_id == 3:
                is_counter = -1
        elif self.element_id == 3:
            if op_character.element_id == 2:
                is_counter = 1
            elif op_character.element_id == 4:
                is_counter = -1
        elif self.element_id == 4:
            if op_character.element_id == 3:
                is_counter = 1
            elif op_character.element_id == 1:
                is_counter = -1

            
        '''
        is_counter = 0
        if (self.element_id - op_character.element_id) % 2 != 0:
            if self.element_id - op_character.element_id == 1 or self.element_id - op_character.element_id == -3:
                is_counter = 1
            else:
                is_counter = -1
        '''
        #攻击力的计算
        attack = base_attack_value
        if is_counter == 1:
            pause(pause_t)
            print("属性伤害翻倍")
            attack *= 2
        elif is_counter == -1:
            pause(pause_t)
            print("属性伤害减半")
            attack /= 2
        attack = max(0,attack - op_character.defend_value)
        if op_character.defend_rate:
            pause(pause_t)
            print(f"护盾吸收 {op_character.defend_rate}%")
            attack *= (100 - op_character.defend_rate)
            attack /= 100
        if is_dodge:
            attack = 0
        if op_character.is_fired:
            pause(pause_t)
            print(f"{op_character.belong} {op_character.chinese} 受到 10 点额外烧伤 {3 - op_character.is_fired * 2}/2")
            attack += 10
            op_character.is_fired -= 0.5
        if attack != 0:
            if self.element_id == 2:
                attack *= 1 + 0.1 * self.passive_skill()

        return attack
    #伤害扣除
    def reduce_hp(self,num:int,object:Pokemon):
        #水被动
        if num != 0 and object.element_id == 3:
            object.passive_skill(num)
        object.hp -= num
        pause(pause_t)
        print(f"{object.belong} {object.chinese} 受到了 {num} 点伤害！剩余 HP：{object.hp}/{object.max_hp}")
        #电被动
        if num != 0:
            if object.element_id == 4:
                object.passive_skill()


    #生命恢复
    def recover_hp(self,num:int):
        self.hp = min(self.max_hp,self.hp + num)
        pause(pause_t)
        print(f"{self.belong} {self.chinese} 生命值回复 {num}")

    #护盾增加
    def add_defend(self,num:int):
        self.defend_rate=min(100,self.defend_rate + num)
        pause(pause_t)
        print(f"{self.belong} {self.chinese} 护盾增加 {num}，目前护盾: {self.defend_rate}%")

    #闪避率增加
    def add_dodge(self,num:int):
        self.dodge_rate = min (100,self.dodge_rate + num)
        pause(pause_t)
        print(f"{self.belong} {self.chinese} 闪避率增加 {num}，目前闪避值：{self.dodge_rate}")

    #技能
    def skill_act(self,skill_num:int,op_character:Pokemon):
        if self.name == 'PikaChu':
            #十万伏特
            if skill_num == 1:
                self.reduce_hp(1.4 * self.attack(op_character,self.attack_value),op_character)
                #麻痹
                if lucky_bar(0.1,'十万伏特麻痹'):
                #random.random() <= 0.1:
                    pause(pause_t)
                    print(f"{op_character.belong} {op_character.chinese} 已被麻痹")
                    op_character.is_paralyzed = 1

            #电光一闪
            if skill_num == 2:
                self.reduce_hp(self.attack(op_character,self.attack_value),op_character)
                if lucky_bar(0.1,'电光一闪连续攻击'):
                #random.random() <= 0.1:
                    pause(pause_t)
                    print("**触发连续攻击**")
                    self.reduce_hp(self.attack(op_character,self.attack_value),op_character)
        if self.name == 'Bulbasaur':
            #种子炸弹
            if skill_num == 1:
                self.reduce_hp(self.attack(op_character,self.attack_value),op_character)
                #中毒
                if lucky_bar(0.1,'种子炸弹中毒'):
                #random.random() <= 0.1:
                    op_character.is_poisoning = 1
            #寄生种子
            if skill_num == 2:
                self.is_seed = 3
                pause(pause_t)
                print("寄生种子下回合生效")

        if self.name == 'Squirtle':
            #水枪
            if skill_num == 1:
                self.reduce_hp(1.4 * self.attack(op_character,self.attack_value),op_character)

            #护盾
            if skill_num == 2:
                self.add_defend(50)
                self.reduce_hp(self.attack(op_character,0),op_character)

        if self.name == 'Charmander':
            #火花
            if skill_num == 1:
                self.reduce_hp(self.attack(op_character,self.attack_value),op_character)
                #烧伤
                if lucky_bar(0.1,'烧伤'):
                #random.random() <= 0.1:
                    pause(pause_t)
                    print(f"{op_character.belong} {op_character.chinese} 成功被烧伤")
                    op_character.is_fired = 1
                
            #蓄能爆炎
            if skill_num == 2:
                if self.flame_count:
                    op_character.add_dodge(20)
                    self.reduce_hp(3 * self.attack(op_character,self.attack_value),op_character)
                    #烧伤
                    if lucky_bar(0.8,'烧伤'):
                    #random.random() < 0.8:
                        pause(pause_t)
                        print(f"{op_character.belong} {op_character.chinese} 成功被烧伤")
                        op_character.is_fired = 1
                    self.flame_count = 0
                else:
                    self.flame_count += 1
                    pause(pause_t)
                    print(f"{self.belong} {self.chinese}蓄能爆炎 已蓄能！")

            

#元素类
class WaterPokemon(Pokemon):
    def __init__(self, name, hp, attack_value, defend_value, dodge_rate,chinese):
        super().__init__(name, hp, attack_value, defend_value, dodge_rate,chinese)

    #类属性
    element = 'water'
    element_id = 3
    element_chinese = '水属性'
    def passive_skill(self,attack:int):
        if lucky_bar(0.5,'水属性被动'):
        #random.random() <= 0.5:
            pause(pause_t)
            print("水属性被动发动，伤害减免30%")
            attack *= 0.7

class FirePokemon(Pokemon):
    def __init__(self, name, hp, attack_value, defend_value, dodge_rate,chinese):
        super().__init__(name, hp, attack_value, defend_value, dodge_rate,chinese)
        self.passive_skill_num = 0
    #类属性
    element = 'fire'
    element_id = 2
    element_chinese = '火属性'
    
    def passive_skill(self)->int:
        pause(pause_t)
        print(f"{self.chinese} 被动启用")
        if self.passive_skill_num <= 4:
            self.passive_skill_num += 1
        print(f"当前层数 （{self.passive_skill_num}/4）")
        return self.passive_skill_num

class GrassPokemon(Pokemon):
    def __init__(self, name, hp, attack_value, defend_value, dodge_rate,chinese):
        super().__init__(name, hp, attack_value, defend_value, dodge_rate,chinese)
    #类属性
    element = 'grass'
    element_id = 1
    element_chinese = '草属性'
    def passive_skill(self):
        pause(pause_t)
        print(f"{self.chinese} 回合触发草属性被动")
        self.recover_hp(0.1 * self.max_hp)
class ElectricPokemon(Pokemon):
    def __init__(self, name, hp, attack_value, defend_value, dodge_rate,chinese):
        super().__init__(name, hp, attack_value, defend_value, dodge_rate,chinese)
        self.is_act = 0
    #类属性
    element = 'electric'
    element_id = 4
    element_chinese = '电属性'
    def passive_skill(self):
        self.is_act = 1