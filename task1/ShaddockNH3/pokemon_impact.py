import copy
import random
import sys

class DodgeException(Exception):
    pass

class Pokemon:
    def __init__(self,hp,attack,defence,dodge):
        self.hp=hp
        self.attack=attack
        self.defence=defence
        self.dodge=dodge
        self.skills=[]
        self.skill_data=None
        self.alive=True
        self.initial_hp=hp
        self.now_turn=1
        self.state=[]
        self.is_dodge=False

    '''
    def has_dodged(self,probability):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if random.random() < probability:
                    print("你的技能被闪避！")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    #上述装饰器备用
    '''

    #也可以使用抛出异常模块，简单判断的代码感觉过于简单了，而且不好改
    #装饰器也不好改
    def try_dodge(self):
        if random.random()<=self.dodge:
            raise DodgeException("技能被闪避！")

    def receive_damage(self, damage,pokemon,other_pokemon):
        if pokemon.hp-damage<=0:
            pokemon.hp=0
        else:
            pokemon.hp-=damage

    def receive_normal_damage(self,damage,pokemon,other_pokemon):
        if pokemon is None:
            pokemon = self
        if damage-pokemon.defence<=0:
            damage=0
        else:
            damage-=pokemon.defence
        pokemon.hp-=damage
        print(f"该伤害为普通物理伤害，对方防御{pokemon.defence}")

    def receive_restore(self, restore, pokemon=None):
        if pokemon is None:
            pokemon=self

        if pokemon.hp+restore>=pokemon.initial_hp:
            pokemon.hp=pokemon.initial_hp
        else:
            pokemon.hp += restore

    def restraint(self, other_pokemon,property_1,property_2):
        if other_pokemon.property == property_1:
            print("克制对方宝可梦")
            return self.attack * 1.5
        elif other_pokemon.property == property_2:
            print("被对方宝可梦克制")
            return self.attack * 0.5
        else:
            return self.attack

    def useskill(self,func,other_pokemon):
        skill_func=getattr(self.skill_data,func.__name__)
        skill_func(other_pokemon,self)

    def print_damage_happened(self,other_pokemon,now_damage,a_skill):
        print(f"{self.name} 使用了 {a_skill}")
        print(f"{other_pokemon.name} 受到了 {now_damage} 点伤害！剩余 HP：{other_pokemon.hp}")

    def print_restore_happened(self, now_restore, a_skill):
        print(f"{self.name} 使用了 {a_skill}")
        print(f"{self.name} 恢复了 {now_restore} 点伤害！ 剩余 HP：{self.hp}")

class Grass_Pokemon(Pokemon):#草属性宝可梦
    def __init__(self,hp,attack,defence,dodge):
        Pokemon.__init__(self,hp,attack,defence,dodge)
        self.property = "grass"

    def restraint_grass(self, other_pokemon):
        return self.restraint(other_pokemon, "water", "fire")

    def begin(self):
        now_restore=self.initial_hp*0.1
        self.receive_restore(now_restore)
        print("草属性被动使用成功")
        print("宝可梦当前血量",end=" ")
        print(self.hp)

    def end(self):
        pass


class Fire_Pokemon(Pokemon):
    def __init__(self,hp,attack,defence,dodge):
        Pokemon.__init__(self,hp,attack,defence,dodge)
        self.property="fire"
        self.is_attack=False
        self.index=1

    def restraint_fire(self,other_pokemon):
        return self.restraint(other_pokemon, "grass", "water")

    def begin(self):
        if self.index<=4 and self.is_attack:
            self.attack *= 1.1
            self.index+=1
            print("火属性被动使用成功,造成伤害基础攻击力*1.1倍，最多4层")
            self.is_attack=False


    def end(self):
        pass

class Water_Pokemon(Pokemon):#水属性宝可梦
    def __init__(self,hp,attack,defence,dodge):#继承
        Pokemon.__init__(self,hp,attack,defence,dodge)
        self.property="water"

    def restraint_water(self,other_pokemon):
        return self.restraint(other_pokemon,"fire","eletri")

    def receive_damage(self, damage,pokemon,other_pokemon):
        if random.random()<=0.5:
            damage*=0.7
            print("水属性被动使用成功")

        if pokemon.hp-damage<=0:
            pokemon.hp=0
        else:
            pokemon.hp-=damage

    def receive_normal_damage(self,damage,pokemon,other_pokemon):
        if pokemon is None:
            pokemon = self
        if random.random()<=0.5:
            damage*=0.7
            print("水属性被动使用成功")

        if damage-pokemon.defence<=0:
            damage=0
        else:
            damage-=pokemon.defence
        pokemon.hp-=damage
        print(f"该伤害为普通物理伤害，对方防御{pokemon.defence}")

    def begin(self):
        pass

    def end(self):
        pass

class Eletri_Pokemon(Pokemon):
    def __init__(self,hp,attack,defence,dodge):
        Pokemon.__init__(self,hp,attack,defence,dodge)
        self.property = "eletri"

    def restraint_eletri(self, other_pokemon):
        return self.restraint(other_pokemon, "water", "grass")

    def try_dodge(self):
        if random.random()<=self.dodge:
            self.is_dodge=True
            raise DodgeException("对方是雷属性，技能被对方闪避，对方立刻使用一个技能！")

    def begin(self):
        pass

    def end(self):
        #这里增加，如果触发了被动，那么可以立刻使用一次技能
        pass

class Ice_Pokemon(Pokemon):
    def __init__(self,hp,attack,defence,dodge):
        Pokemon.__init__(self,hp,attack,defence,dodge)
        self.property = "ice"
        self.index=1

    #冰系克制草，被火克制
    def restraint_ice(self,other_pokemon):
        return self.restraint(other_pokemon,"grass","fire")

    '''
    受到攻击时，对手的攻击力下降5点，最多叠加3层
    '''

    def forst(self,other_pokemon):
        if self.index<=3:
            self.index+=1
            print("对方冰属性被动使用成功,当受到攻击时，我方宝可梦攻击力下降5点，最多三次")
            other_pokemon.attack-=5

    def receive_damage(self, damage,pokemon,other_pokemon):
        if pokemon.hp - damage <= 0:
            pokemon.hp = 0
        else:
            pokemon.hp -= damage
        self.forst(other_pokemon)

    def receive_normal_damage(self, damage, pokemon,other_pokemon):

        if damage - pokemon.defence <= 0:
            damage = 0
        else:
            damage -= pokemon.defence
        pokemon.hp -= damage
        print(f"该伤害为普通物理伤害，对方防御{pokemon.defence}")

        self.forst(other_pokemon)

    def begin(self):
        pass

    def end(self):
        pass


class Dragon_Pokemon(Pokemon):
    def __init__(self, hp, attack, defence, dodge):
        Pokemon.__init__(self, hp, attack, defence, dodge)
        self.property = "dragon"
        self.is_draconic_will=False

    # 龙系克制龙，被龙克制
    def restraint_dragon(self, other_pokemon):
        return self.restraint(other_pokemon, "dragon", "dragon")

    '''
    龙属性特性: 龙魂（Draconic Will）
    当我方生命降低至50%及以下时，在回合结束后，攻击力增加10点，防御力增加5点（每回合）
    '''

    def draconic_will(self):
        if self.hp<=0.5*self.initial_hp:
            self.is_draconic_will=True

    def begin(self):
        pass

    def end(self):
        self.draconic_will()
        if self.is_draconic_will:
            self.attack+=10
            self.dodge+=5
            self.is_draconic_will=False
            print("龙属性被动使用成功")

class Pokemon_PikaChu(Eletri_Pokemon):
    def __init__(self):
        Eletri_Pokemon.__init__(self,hp=80,attack=35,defence=5,dodge=0.3)
        self.name="PikaChu"

        self.skill_data=PikaChu_skills()
        self.skill_data.add_skills_to_list()
        self.skills=self.skill_data.skill_list

class Pokemon_Bulbasuar(Grass_Pokemon):
    def __init__(self):
        Grass_Pokemon.__init__(self,hp=100,attack=35,defence=10,dodge=0.3)
        self.name="Bulbasuar"
        self.skill_data=Bulbasaur_skills()
        self.skill_data.add_skills_to_list()
        self.skills=self.skill_data.skill_list

class Pokemon_Squirtle(Water_Pokemon):
    def __init__(self):
        Water_Pokemon.__init__(self,hp=80,attack=25,defence=20,dodge=0.2)
        self.name="Squirtle"
        self.skill_data=Squirtle_skills()
        self.skill_data.add_skills_to_list()
        self.shield=False
        self.skills=self.skill_data.skill_list

    def change_shield(self):
        self.shield=True

    def receive_damage(self, damage,pokemon,other_pokemon):
        if random.random()<=0.5:
            damage*=0.7

        if self.shield is True:
            damage*=0.5
            self.shield=False

        if pokemon.hp-damage<=0:
            pokemon.hp=0
        else:
            pokemon.hp-=damage


class Pokemon_Charmander(Fire_Pokemon):
    def __init__(self):
        Fire_Pokemon.__init__(self,hp=80,attack=35,defence=15,dodge=0.1)
        self.name="Charmander"
        self.skill_data=Charmander_skills()
        self.skill_data.add_skills_to_list()
        self.flame_charge=False
        self.skills=self.skill_data.skill_list

    def is_flame_charge(self):
        return self.flame_charge

    def change_flame_charge(self):
        self.flame_charge=not self.flame_charge

class Pokemon_Galcidrake(Ice_Pokemon,Dragon_Pokemon):
    def __init__(self):
        Ice_Pokemon.__init__(self, hp=110, attack=30, defence=8, dodge=0.12)
        Dragon_Pokemon.__init__(self,hp=110,attack=30,defence=8,dodge=0.12)
        self.name = "Galcidrake"
        self.skill_data = Galcidrake_skills()
        self.skill_data.add_skills_to_list()
        self.skills = self.skill_data.skill_list

'''
自己设计的宝可梦
双属性宝可梦！
龙、冰
属性: Ice/Dragon
基础数值:
血量 (HP): 110
攻击力 (Attack): 30
防御力 (Defense): 8
闪避率 (Dodge Rate): 12%
'''

def create_pokemon_list():
    pokemon_pikachu=Pokemon_PikaChu()
    pokemon_bulbasuar=Pokemon_Bulbasuar()
    pokemon_squirtle=Pokemon_Squirtle()
    pokemon_charmander=Pokemon_Charmander()
    pokemon_galcidrake=Pokemon_Galcidrake()

    pokemon_list=[]

    pokemon_list.append(pokemon_pikachu)
    pokemon_list.append(pokemon_bulbasuar)
    pokemon_list.append(pokemon_squirtle)
    pokemon_list.append(pokemon_charmander)
    pokemon_list.append(pokemon_galcidrake)

    return pokemon_list

class Skills:

    def __init__(self):
        self.skill_list=[]

    '''
    def show_skill(self):
        i=1
        for skill in self.skill_list:
            print(f"{i}:{skill.__name__}")
            i+=1
    '''

    def add_skill(self,skill):
        self.skill_list.append(skill)

    @classmethod
    def add_is_skill(self,func):
        func.is_skill=True
        return func

    def add_skills_to_list(self):
        for name,obj in self.__class__.__dict__.items():
            if callable(obj) and hasattr(obj,"is_skill") and obj.is_skill:
                self.add_skill(obj)

class PikaChu_skills(Skills):

    def __init__(self):
        super().__init__()

    @Skills.add_is_skill
    def Thunderbolt(self,other_pokemon,pokemon):
        try:
            other_pokemon.try_dodge()

            now_damage=1.4*pokemon.restraint_eletri(other_pokemon)
            other_pokemon.receive_damage(now_damage,other_pokemon,pokemon)
            pokemon.print_damage_happened(other_pokemon, now_damage, "Thunderbolt")

            #麻痹效果
            if random.random()<=0.1:
                numb_effect = Numb_Effect(1)
                pokemon.state.append(numb_effect)

        except DodgeException as e:
            print(e)

    @Skills.add_is_skill
    def Quick_Attack(self,other_pokemon,pokemon):
        try:
            other_pokemon.try_dodge()

            now_damage=pokemon.attack
            other_pokemon.receive_normal_damage(now_damage, other_pokemon,pokemon)
            pokemon.print_damage_happened(other_pokemon, now_damage, "Quick_Attack")
            if random.random()<=0.1:
                print("触发第二次攻击")
                try:
                    other_pokemon.try_dodge()

                    other_pokemon.receive_normal_damage(now_damage, other_pokemon,pokemon)
                    pokemon.print_damage_happened(other_pokemon, now_damage, "Quick_Attack")


                except DodgeException as e:
                    print(e)

        except DodgeException as e:
            print(e)

class Bulbasaur_skills(Skills):
    def __init__(self):
        super().__init__()

    @Skills.add_is_skill
    def Seed_Bomb(self, other_pokemon,pokemon):
        try:
            other_pokemon.try_dodge()

            #这里补充，草属性伤害十点。
            now_damage=pokemon.restraint_grass(other_pokemon)
            other_pokemon.receive_damage(now_damage,other_pokemon,pokemon)
            pokemon.print_damage_happened(other_pokemon, now_damage, "Seed_Bomb")
            #这里需要实现中毒效果

            if random.random()<=0.15:
                poision_effect=Poision_Effect(5)
                other_pokemon.state.append(poision_effect)

            #以下是斩杀
            if other_pokemon.hp<=20:
                other_pokemon.hp=0

        except DodgeException as e:
            print(e)

    @Skills.add_is_skill
    def Parasitic_Seeds(self, other_pokemon,pokemon=None):
        #这里需要实现自动调用三回合
        p_s_e=Parasitic_Seed_Effect(3)
        other_pokemon.state.append(p_s_e)
        print("Parasitic_Seed使用成功，为特殊效果")


class Squirtle_skills(Skills):
    def __init__(self):
        super().__init__()

    @Skills.add_is_skill
    def Aqua_Jet(self, other_pokemon,pokemon=None):
        try:
            other_pokemon.try_dodge()

            now_damage = 1.4 * pokemon.restraint_water(other_pokemon)
            other_pokemon.receive_damage(now_damage, other_pokemon,pokemon)
            pokemon.print_damage_happened(other_pokemon, now_damage, "Aqua_Jet")

        except DodgeException as e:
            print(e)

    @Skills.add_is_skill
    def Shield(self, other_pokemon,pokemon):

        pokemon.change_shield()
        print("Squirtle 使用了 Shield",end=" ")
        print("减少下一回合受到的伤害50%")


class Charmander_skills(Skills):
    def __init__(self):
        super().__init__()

    @Skills.add_is_skill
    def Ember(self, other_pokemon,pokemon):
        if not pokemon.is_flame_charge():
            try:
                other_pokemon.try_dodge()
                now_damage = pokemon.restraint_fire(other_pokemon)
                other_pokemon.receive_damage(now_damage, other_pokemon,pokemon)
                pokemon.print_damage_happened(other_pokemon, now_damage, "Ember")
                if random.random() <= 0.1:
                    #烧伤
                    burn_effect = Burn_Effect(2, 10)
                    other_pokemon.state.append(burn_effect)

                    pokemon.is_attack=True
            except DodgeException as e:
                print(e)

        else:
            print("你2技能正在蓄力中，自动为您跳转到2技能的实现！")
            self.Flame_Charge(other_pokemon,pokemon)


    @Skills.add_is_skill
    def Flame_Charge(self, other_pokemon,pokemon):
        if not pokemon.is_flame_charge():
            pokemon.change_flame_charge()
            # 这里需要实现一回合蓄力,用了一个笨方法暂时
            print("使用flame_charge,需要一回合蓄力")

        else:
            pokemon.change_flame_charge()

            now_damage=3*pokemon.restraint_fire(other_pokemon)
            other_pokemon.dodge+=0.2
            print("蓄力成功")

            try:
                other_pokemon.try_dodge()

                if random.random() <= 0.8:
                    burn_effect = Burn_Effect(2, 10)
                    other_pokemon.state.append(burn_effect)
                    # 烧伤状态

                other_pokemon.receive_damage(now_damage, other_pokemon,pokemon)
                pokemon.print_damage_happened(other_pokemon, now_damage, "Flame_Charge")

                pokemon.is_attack = True

            except DodgeException as e:
                print(e)

            other_pokemon.dodge -= 0.2

class Galcidrake_skills(Skills):
    def __init__(self):
        super().__init__()
        self.glacial_surge_cold=0

    @Skills.add_is_skill
    def Frozen_Fang(self,other_pokemon,pokemon):
        try:
            other_pokemon.try_dodge()

            now_damage=pokemon.attack*1.6
            other_pokemon.receive_damage(now_damage,other_pokemon,pokemon)
            pokemon.print_damage_happened(other_pokemon,now_damage,"Frozen_Fang")
            if random.random()<=0.3:
                if other_pokemon.attack-5<=0:
                    other_pokemon.attack=10
                else:
                    other_pokemon.attack-=5
                print("对方宝可梦陷入龙惧状态，攻击力减5，最多减少到10")

        except DodgeException as e:
            print(e)

        if self.glacial_surge_cold>0:
            self.glacial_surge_cold-=1


    @Skills.add_is_skill
    def Glacial_Surge(self,other_pokemon,pokemon):
        if self.glacial_surge_cold!=0:
            print(f"这个技能正在冷却中，还剩下{self.glacial_surge_cold}回合后才能使用,自动跳转至frozen_fang")
            self.Frozen_Fang(other_pokemon,pokemon)
        else:
            try:
                other_pokemon.try_dodge()

                now_damage=0.7*pokemon.restraint_ice(other_pokemon)
                other_pokemon.receive_damage(now_damage,other_pokemon,pokemon)
                other_pokemon.dodge-=0.05
                pokemon.dodge+=0.05
                pokemon.receive_restore(30)
                pokemon.print_damage_happened(other_pokemon,now_damage,"Glacial_Surge")
                print("对方闪避率降低5%。Glacidrake的闪避率增加5%，并恢复30的生命")

            except DodgeException as e:
                print(e)
            self.glacial_surge_cold=3

''' 
技能:
1. Frozen Fang
类型: Dragon
效果:
对目标造成攻击力的1.6倍的普通！伤害，并有30%几率使自己陷入“龙惧”状态
龙惧：被自己的技能吓到了，普通攻击力减少5点

2. Glacial Surge
类型: Ice
效果:
召唤一场寒冰风暴，对敌方宝可梦造成0.7倍冰属性伤害，并使它们的闪避率降低5%。
Glacidrake的闪避率增加5%，并恢复30的生命
冷却时间: 3回合。
'''

class Effect:
    name:str

    def __init__(self,duration):
        self.duration=duration

    def decrease_duration_time(self):
        self.duration-=1
        print(f"{self.name} 持续 {self.duration}回合")
        print()

    def apply(self,pokemon,my_pokemon=None):
        pass

    def print_damage(self,pokemon,damage):
        print(f"{pokemon.name} 陷入 {self.name} 状态 受到 {damage} 伤害 当前 hp:{pokemon.hp}")
        print()

    def print_restore(self,pokemon,restore):
        print(f"{pokemon.name} 陷入 {self.name} 状态 恢复 {restore} hp 当前hp:{pokemon.hp}")

class Poision_Effect(Effect):
    name="Poison"

    def __init__(self,duration):
        super().__init__(duration)

    def apply(self,pokemon,my_pokemon=None):
        damage=pokemon.initial_hp*0.1
        pokemon.receive_damage(damage,pokemon,my_pokemon)
        self.print_damage(pokemon,damage)

class Parasitic_Seed_Effect(Poision_Effect):
    name = "Parasitic_Seed"

    def __init__(self, duration, restore=0):
        super().__init__(duration)

    def apply(self, pokemon,my_pokemon=None):
        now_restore=pokemon.initial_hp*0.1
        now_damage=now_restore
        pokemon.receive_damage(now_damage,pokemon,my_pokemon)
        if pokemon.hp+pokemon.initial_hp*0.1>pokemon.initial_hp:
            pokemon.hp=pokemon.initial_hp
        else:
            pokemon.receive_restore(now_restore)
        self.print_damage(pokemon, now_damage)
        self.print_restore(my_pokemon, now_restore)

class Burn_Effect(Effect):
    name="Burn"

    def __init__(self,duration,damage,restore=0):
        super().__init__(duration)
        self.damage=damage

    def apply(self,pokemon,my_pokemon=None):
        pokemon.receive_damage(self.damage,pokemon,my_pokemon)
        self.print_damage(pokemon,self.damage)

class Numb_Effect(Effect):
    name="Numb"

    def __init__(self,duration):
        super().__init__(duration)

    def apply(self,pokemon,my_pokemon=None):
        pokemon.now_turn+=1
        print("触发麻痹，对手跳过一回合")



class Play:
    def __init__(self,all_pokemon,num):
        self.all_pokemon=all_pokemon
        self.player_team=[]
        self.computer_team=[]
        self.current_p_pokemon=None
        self.current_c_pokemon=None
        self.turn=1
        self.num=num

    def p_choose_pokemon_team(self):
    #玩家选择宝可梦列表中的宝可梦
        while True:
            print(f"请选择{self.num}个宝可梦用于组成你的队伍：")
            pokemon_to_choose_replica=copy.deepcopy(self.all_pokemon)
            self.print_list(pokemon_to_choose_replica)
            print("请输入数字，一次性选择你的所有宝可梦。")
            print("要求数字间用空格隔开，可以不按顺序输入，否则将重新输入。")
            num_of_pokemon=list(map(int,input().split()))
            num_of_pokemon.sort()
            if self.is_choose_pokemon_team(num_of_pokemon,pokemon_to_choose_replica):
                self.player_team.extend(pokemon_to_choose_replica[nums - 1] for nums in num_of_pokemon)
                break
            print("你输入了错误的格式！请重新输入！")
        print("你的宝可梦队伍如下")
        self.print_list(self.player_team)

    def is_choose_pokemon_team(self,num_of_pokemon,pokemon_to_choose_replica):
        if (len(num_of_pokemon)==self.num
            and len(num_of_pokemon)==len(set(num_of_pokemon))
            and self.is_choose_in_range(num_of_pokemon[-1],pokemon_to_choose_replica)):
            return True
        #这里代码的逻辑，第一行是检查用户输入的长度是否和给定长度的宝可梦相同
        #第二行是利用集合，确定没有重复数字
        #第三行是确定排序后的最大数字没有越过原始的宝可梦列表长度！
        return False

    def is_choose_in_range(self,num,l):
        if num<=len(l):
            return True
        return False

    def print_list(self,pokemon_list):
        i=1
        for p in pokemon_list:
            print(f"{i}:{p.name}")
            i+=1

    def c_choose_pokemon_team(self):
        pokemon_to_choose_replica = copy.deepcopy(self.all_pokemon)
        num_of_pokemon = sorted(random.sample(range(1, len(pokemon_to_choose_replica)+1), self.num))
        self.computer_team.extend(pokemon_to_choose_replica[nums - 1] for nums in num_of_pokemon)

        print("电脑的宝可梦队伍如下")
        self.print_list(self.computer_team)

    def p_choose_current_pokemon(self):
        print("请选择你的宝可梦出战！")
        self.print_list(self.player_team)
        while True:
            print("请输入数字选择你的宝可梦：",end=" ")
            current_choose=int(input())
            if self.is_choose_in_range(current_choose,self.player_team):
                chosen_pokemon=self.player_team[current_choose-1]
                if chosen_pokemon.alive:  #这里得另外实现
                    self.current_p_pokemon=chosen_pokemon
                    if self.current_c_pokemon is None:
                        self.current_p_pokemon.now_turn=self.turn
                    else:
                        self.current_p_pokemon.now_turn=self.current_c_pokemon.now_turn
                    return chosen_pokemon
                else:
                    print("这只宝可梦昏厥了，请重新选择")
            else:
                print("你没有这个宝可梦！")

    def c_chosse_current_pokemon(self):
        c_alive_team=[pokemon for pokemon in self.computer_team if pokemon.alive]
        self.current_c_pokemon=random.choice(c_alive_team)
        print("电脑选择了",end=" ")
        print(self.current_c_pokemon.name)
        self.current_c_pokemon.now_turn=self.turn
        return self.current_c_pokemon

    def p_chosse_skills(self):
        print(f"你的{self.current_p_pokemon.name}的技能如下：")
        skills=self.current_p_pokemon.skills#这里是宝可梦技能类实现
        i=1
        for skill in skills:
            print(f"{i}:{skill.__name__}")
            i+=1

        while True:
            print("请选择一个技能进行攻击：")
            skills_num = int(input())
            if self.is_choose_in_range(skills_num,skills):
                self.current_p_pokemon.useskill(skills[skills_num-1],self.current_c_pokemon)#这里也要另外实现
                break
            else:
                print("技能范围越界，请重新选择技能")

    def c_choose_skills(self):
        skills=self.current_c_pokemon.skills
        skill_option=random.choice(skills)
        self.current_c_pokemon.useskill(skill_option,self.current_p_pokemon)

    def finish_game(self):
        print()
        input("输入任意值后结束进程")
        sys.exit()
        #强行结束进程

    def check_game_status(self):
        if self.current_p_pokemon.hp<=0:
            self.current_p_pokemon.alive=False

        if self.current_c_pokemon.hp<=0:
            self.current_c_pokemon.alive=False

        #检查游戏状态，判断游戏是应该是结束，还是继续选择新的宝可梦！
        is_player_fail=all(pokemon.alive is False for pokemon in self.player_team)
        is_computer_fail=all(pokemon.alive is False for pokemon in self.computer_team)

        if is_player_fail and is_computer_fail:
            print("双方宝可梦全部都昏厥了，平局")
            self.finish_game()

        elif is_player_fail and not is_computer_fail:
            print("你输了！电脑赢了！")
            self.finish_game()

        elif not is_player_fail and is_computer_fail:
            print("你赢了！电脑输了！")
            self.finish_game()

        else:
            if not self.current_p_pokemon.alive:
                self.p_choose_current_pokemon()
            if not self.current_c_pokemon.alive:
                self.c_chosse_current_pokemon()

    def a_battle_begin(self):
        print(f"这是第{self.turn}回合")

        print("玩家回合开始！")
        self.current_p_pokemon.begin()#有些属性宝可梦可能会在开局的时候就触发某些状态！
        self.check_game_status()

        print("电脑回合开始！")
        self.current_c_pokemon.begin()
        self.check_game_status()

    def a_battle_now(self):
        print("这是玩家的回合")
        if self.current_p_pokemon.now_turn!=self.turn:
            print("此回合，玩家跳过！")
        else:
            print("玩家使用技能")
            self.p_chosse_skills()
            self.current_p_pokemon.now_turn+=1

            self.check_game_status()

            if self.current_c_pokemon.is_dodge:
                print("立刻多一回合")
                self.c_choose_skills()
                self.current_c_pokemon.is_dodge=False

        self.check_game_status()
        print()

        print("这是电脑的回合")
        if self.current_c_pokemon.now_turn!=self.turn:
            print("此回合，电脑跳过！")
        else:
            print("电脑使用技能")
            self.c_choose_skills()
            self.current_c_pokemon.now_turn+=1

            self.check_game_status()

            if self.current_p_pokemon.is_dodge:
                print("立刻多一回合")
                self.p_chosse_skills()
                self.current_p_pokemon.is_dodge=False

        print()
        self.turn+=1
        self.check_game_status()

    def a_battle_end(self):
        print("玩家回合结束")
        self.current_p_pokemon.end()
        self.check_game_status()

        self.current_p_pokemon_effects=\
            [effect for effect in self.current_p_pokemon.state if effect.duration > 0]

        unique_p_effect={}
        for effect in self.current_p_pokemon_effects:
            if (effect.name not in unique_p_effect or
                effect.duration>unique_p_effect[effect.name].duration):
                unique_p_effect[effect.name]=effect

        self.current_p_pokemon_effects=list(unique_p_effect.values())

        for effect in self.current_p_pokemon_effects:
            effect.decrease_duration_time()
            effect.apply(self.current_c_pokemon,self.current_p_pokemon)
            self.check_game_status()

        print("电脑回合结束")
        self.current_c_pokemon.end()
        self.check_game_status()

        self.current_c_pokemon_effects = \
            [effect for effect in self.current_c_pokemon.state if effect.duration > 0]

        unique_c_effect = {}
        for effect in self.current_c_pokemon_effects:
            if (effect.name not in unique_c_effect or
                    effect.duration > unique_c_effect[effect.name].duration):
                unique_c_effect[effect.name] = effect

        self.current_c_pokemon_effects = list(unique_c_effect.values())

        for effect in self.current_c_pokemon_effects:
            effect.apply(self.current_p_pokemon,self.current_c_pokemon)
            effect.decrease_duration_time()
            self.check_game_status()

        self.current_p_pokemon.state=self.current_p_pokemon_effects
        self.current_c_pokemon.state=self.current_c_pokemon_effects

        print()

    def a_game(self):

        self.p_choose_pokemon_team()
        print()
        self.c_choose_pokemon_team()
        print()

        self.p_choose_current_pokemon()
        print()

        self.c_chosse_current_pokemon()
        print()

        while True:
            self.a_battle_begin()
            self.a_battle_now()
            self.a_battle_end()

if __name__=="__main__":
    pokemon_list=create_pokemon_list()
    print(f"请输入你想玩几个宝可梦一队，最多{len(pokemon_list)}个")
    num=int(input("请输入数字："))
    if 0<num<=len(pokemon_list):
        play=Play(pokemon_list,num)
        play.a_game()
    else:
        print("选择失败，请重新开始选择")
        input("输入任意值后结束进程")
