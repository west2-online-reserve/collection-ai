import random
import sys
import time

class Skill:
    def __init__(self):
        pass

    def use_skill(self,pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        raise NotImplementedError
        
#十万伏特
class Thunderbolt(Skill):
    def __init__(self,base_damage,name = "Thunderbolt",type = "Electric",damage_multiple = 1.4,palsy_rate = 10):
        super().__init__()
        self.name = name
        self.type = type
        self.base_damage = base_damage
        self.damage_multiple = damage_multiple
        self.palsy_rate = palsy_rate

    def check_type(self,enemy_pokemon:"Pokemon"):
        if enemy_pokemon.type == "Water":
            return 2
        elif enemy_pokemon.type == "Grass":
            return 0.5
        else:
            return 1
        
    def use_skill(self,pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        type_multiple = self.check_type(enemy_pokemon)
        self.final_damage = self.base_damage * self.damage_multiple * type_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:  
            enemy_pokemon.receive_damage(self.final_damage)
            palsy_number = random.randint(1,100)
            if palsy_number <= self.palsy_rate:
                print(f"{enemy_pokemon.name} is palsied!")                                                                                     
                enemy_pokemon.statuses.append(Palsy())

    def __str__(self):
        return "Thunderbolt"

#电光一闪
class Quick_Attack(Skill):
    def __init__(self,base_damage,name = "Quick Attack",damage_multiple = 1,second_rate = 10):
        super().__init__()
        self.name = name
        self.base_damage = base_damage
        self.damage_multiple = damage_multiple
        self.second_rate = second_rate

    def use_skill(self, pokemon:"Pokemon", enemy_pokemon:"Pokemon"):
        self.final_damage = self.base_damage * self.damage_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:   
            enemy_pokemon.receive_damage(self.final_damage)
            second_number = random.randint(1,100)
            if second_number <= self.second_rate:                                                                                       
                print(f"{pokemon.name} triggered the second attack!")
                self.use_skill(pokemon,enemy_pokemon)

    def __str__(self):
        return "Quick Attack"

#种子炸弹
class Seed_Bomb(Skill):
    def __init__(self,base_damage,name = "Seed Bomb",type = "Grass",poison_rate = 15):
        super().__init__()
        self.name = name
        self.type = type
        self.base_damage = base_damage
        self.poison_rate = poison_rate

    def check_type(self,enemy_pokemon:"Pokemon"):
        if enemy_pokemon.type == "Water":
            return 2
        elif enemy_pokemon.type == "Fire":
            return 0.5
        else:
            return 1
        
    def use_skill(self,pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        type_multiple = self.check_type(enemy_pokemon)
        self.final_damage = self.base_damage * type_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:   
            enemy_pokemon.receive_damage(self.final_damage)
            poison_number = random.randint(1,100)
            if poison_number <= self.poison_rate:
                print(f"{enemy_pokemon.name} is poisoned!")                                                                   
                enemy_pokemon.statuses.append(Poison())

    def __str__(self):
        return "Seed Bomb"

#寄生种子
class Parasitic_Seeds(Skill):
    def __init__(self,name = "Parasitic Seeds",absorb_rate = 0.1):
        super().__init__()
        self.name = name
        self.absorb_rate = absorb_rate

    def use_skill(self, pokemon:"Pokemon", enemy_pokemon:"Pokemon"):                                                       
        pokemon.statuses.append(Parasitic())
        pokemon.apply_status(Parasitic,enemy_pokemon)

    def __str__(self):
        return "Parasitic Seeds"

#水枪
class Aqua_Jet(Skill):
    def __init__(self,base_damage,name = "Aqua Jet",type = "Water",damage_multiple = 1.4):
        super().__init__()
        self.name = name
        self.type = type
        self.base_damage = base_damage
        self.damage_multiple = damage_multiple

    def check_type(self,enemy_pokemon:"Pokemon"):
        if enemy_pokemon.type == "Fire":
            return 2
        elif enemy_pokemon.type == "Electric":
            return 0.5
        else:
            return 1
        
    def use_skill(self, pokemon:"Pokemon", enemy_pokemon:"Pokemon"):
        type_multiple = self.check_type(enemy_pokemon)
        self.final_damage = self.base_damage * type_multiple * self.damage_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:
            enemy_pokemon.receive_damage(self.final_damage)

    def __str__(self):
        return "Aqua Jet"

#护盾
class Shield(Skill):
    def __init__(self,name = "Shield",decrease_damage_rate = 0.5):
        super().__init__()
        self.name = name
        self.decrease_damage_rate = decrease_damage_rate

    def use_skill(self, pokemon:"Pokemon",enemy_pokemon:"Pokemon"):                                                
        pokemon.statuses.append(ShieldEffect())

    def __str__(self):
        return "Shield"

#火花
class Ember(Skill):
    def __init__(self,base_damage,name = "Ember",type = "Fire",damage_multiple = 1,burn_rate = 10):
        super().__init__()
        self.name = name
        self.type = type
        self.base_damage = base_damage
        self.damage_multiple = damage_multiple
        self.burn_rate = burn_rate

    def check_type(self,enemy_pokemon:"Pokemon"):
        if enemy_pokemon.type == "Grass":
            return 2
        elif enemy_pokemon.type == "Water":
            return 0.5
        else:
            return 1
    
    def use_skill(self, pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        type_multiple = self.check_type(enemy_pokemon)
        self.final_damage = self.base_damage * self.damage_multiple * type_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:
            enemy_pokemon.receive_damage(self.final_damage)
            burn_number = random.randint(1,100)
            if burn_number <= self.burn_rate:  
                print(f"{enemy_pokemon.name} is burned!")                                                                     
                enemy_pokemon.statuses.append(Burn())
    
    def __str__(self):
        return "Ember"

#蓄能爆炎
class Flame_Charge(Skill):
    def __init__(self,base_damage,name = "Flame Charge",type = "Fire",damage_multiple = 3,burn_rate = 80,enemy_dodge_add = 20):
        super().__init__()
        self.name = name
        self.type = type
        self.base_damage = base_damage
        self.damage_multiple = damage_multiple
        self.burn_rate = burn_rate
        self.enemy_dodge_add = enemy_dodge_add

    def check_type(self,enemy_pokemon:"Pokemon"):
        if enemy_pokemon.type == "Grass":
            return 2
        elif enemy_pokemon.type == "Water":
            return 0.5
        else:
            return 1
    
    def use_skill(self, pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        type_multiple = self.check_type(enemy_pokemon)
        self.final_damage = self.base_damage * type_multiple * self.damage_multiple
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:
            enemy_pokemon.receive_damage(self.final_damage)
            burn_number = random.randint(1,100)
            if burn_number <= self.burn_rate:                                                                   
                print(f"{enemy_pokemon.name} is burned!")
                enemy_pokemon.statuses.append(Burn())

    def __str__(self):
        return "Flame Charge"

class Effect:
    def __init__(self,name,duration):
        self.name = name
        self.duration = duration

    def apply(self,pokemon:"Pokemon"):
        raise NotImplementedError
    
    def duration_decrease(self):
        self.duration -= 1

#麻痹效果
class Palsy(Effect):
    def __init__(self,name = "palsy",duration = 1):
        super().__init__(name,duration)
        self.name = name
        self.duration = duration

    def apply(self,pokemon:"Pokemon"):
        pokemon.palsy_effect = True

#中毒效果
class Poison(Effect):
    def __init__(self,name = "poison",poison_rate = 0.1,duration = 999):
        super().__init__(name,duration)
        self.name = name
        self.poison_rate = poison_rate
        self.duration = duration

    def apply(self,pokemon:"Pokemon"):
        damage = pokemon.HP * self.poison_rate
        pokemon.receive_damage(damage)

#寄生种子效果
class Parasitic(Effect):
    def __init__(self,name = "parasitic",parasitic_rate = 0.1,duration = 3):
        super().__init__(name,duration)
        self.name = name
        self.parasitic_rate = parasitic_rate
        self.duration = duration

    def apply(self,pokemon:"Pokemon",enemy_pokemon:"Pokemon"):
        absorb_amount = enemy_pokemon.max_HP * self.parasitic_rate
        pokemon.heal(absorb_amount)
        if enemy_pokemon.dodge() == True:
            print(f"{enemy_pokemon.name} dodged the attack!")
        elif enemy_pokemon.dodge() == False:
            enemy_pokemon.receive_damage(absorb_amount)
        self.duration_decrease()

#烧伤效果
class Burn(Effect):
    def __init__(self,name = "burn",burn_damage = 10,duration = 2):
        super().__init__(name,duration)
        self.name = name
        self.burn_damage = burn_damage
        self.duration = duration

    def apply(self,pokemon:"Pokemon"):
        pokemon.receive_damage(self.burn_damage)

#护盾效果
class ShieldEffect(Effect):
    def __init__(self,name = "shield",duration = 1):
        super().__init__(name,duration)
        self.name = name
        self.duration = duration

    def apply(self,pokemon:"Pokemon"):
        pokemon.shield_effect = True


class Pokemon:
    def __init__(self,name,type,HP:int,attack:int,defence:int,dodge_rate:int):
        self.name = name
        self.type = type
        self.HP = HP
        self.max_HP = HP
        self.attack = attack
        self.defence = defence
        self.dodge_rate = dodge_rate                #闪避率
        self.alive = True                           #是否存活
        self.statuses = []                          #效果
        self.attack_bonus = 0                       #叠加的攻击层数
        self.shield_effect = False                  #护盾效果
        self.palsy_effect = False                   #麻痹效果
        
    def receive_damage(self,damage):
        if not isinstance(damage,int):
            damage = int(damage)
        if self.shield_effect == True:
            damage *= 0.5
        damage -= self.defence
        if damage <= 0:
            print(f"{self.name}'s defence absorbed the attack!")
        else:
            if self.type == "Water":
                water_passive_number = random.randint(1,100)
                if water_passive_number <= 50:
                    damage *= 0.7
            if not isinstance(self.HP,int):
                self.HP = int(self.HP)
            self.HP -= damage
            if self.HP > 0:
                print(f"{self.name} received {damage} damage. Remaining HP: {self.HP}\n")
            else:
                self.alive = False
                print(f"{self.name} has fainted!")

    def heal(self,heal_amount):
        self.HP += heal_amount
        if self.HP > self.max_HP:
            self.HP = self.max_HP
        print(f"{self.name} has healed {heal_amount}!Remaining HP:{self.HP}")

    def begin(self):
        pass

    def initialize_skill(self):
        raise NotImplementedError
    
    def apply_skill(self,skill:"Skill",enemy_pokemon:"Pokemon"):
        print(f"{self.name} used {skill.name}.")
        skill.use_skill(self,enemy_pokemon)
        if self.type == "Fire" and skill.final_damage - enemy_pokemon.defence > 0:
            if self.attack_bonus == 4:
                print(f"The attack bonus level of {self.name} is already full.")
            elif 0 <= self.attack_bonus < 4:
                self.attack_bonus += 1
                self.attack = self.attack * 1.1


    def add_status(self,status:"Effect"):
        self.statuses.append(status)

    def apply_status(self,status:"Effect",enemy_pokemon:"Pokemon"):
        for status in self.statuses[:]:   
            if isinstance(status,Parasitic):
                status.apply(self,enemy_pokemon)        
            else:
                status.apply(self)
                status.duration_decrease()
            if status.duration <= 0:
                if isinstance(status,ShieldEffect):
                    self.shield_effect = False
                if isinstance(status,Palsy):
                    self.palsy_effect = False
                self.statuses.remove(status)

    def dodge(self):
        dodge_number = random.randint(1,100)
        if dodge_number <= self.dodge_rate:
            return True
        else:
            return False

class GrassPokemon(Pokemon):
    def __init__(self,name,HP,attack,defence,dodge_rate,type = "Grass"):
        super().__init__(name,HP,attack,defence,dodge_rate,type)

    def begin(self):
        self.grass_passive()
        
    def grass_passive(self):
        if not isinstance(self.max_HP,int):
            self.max_HP = int(self.max_HP)
        heal_amount = self.max_HP * 0.1
        self.heal(heal_amount)
        

class WaterPokemon(Pokemon):
    def __init__(self,name,HP,attack,defence,dodge_rate,type = "Water"):
        super().__init__(name,HP,attack,defence,dodge_rate,type)

        
class FirePokemon(Pokemon):
    def __init__(self,name,HP,attack,defence,dodge_rate,type = "Fire"):
        super().__init__(name,HP,attack,defence,dodge_rate,type)

class ElectricPokemon(Pokemon):
    def __init__(self,name,HP,attack,defence,dodge_rate,type = "Electric"):
        super().__init__(name,HP,attack,defence,dodge_rate,type)

#皮卡丘
class Pikachu(ElectricPokemon):
    def __init__(self,name = "Pikachu",type = "Electric",HP = 80,attack = 35,defence = 5,dodge_rate = 30):
        super().__init__(name,type,HP,attack,defence,dodge_rate)

    def initialize_skill(self):
        return [Thunderbolt(base_damage = 35),Quick_Attack(base_damage = 35)]

    def __str__(self):
        return "Pikachu"

#妙蛙种子
class Bulbasaur(GrassPokemon):
    def __init__(self,name = "Bulbasaur",type = "Grass",HP = 100,attack = 35,defence = 10,dodge_rate = 10):
        super().__init__(name,type,HP,attack,defence,dodge_rate)

    def initialize_skill(self):
        return [Seed_Bomb(base_damage = 35),Parasitic_Seeds()]

    def __str__(self):
        return "Bulbasaur"

#杰尼龟   
class Squirtle(WaterPokemon):
    def __init__(self,name = "Squirtle",type = "Water",HP = 80,attack = 25,defence = 20,dodge_rate = 20):
        super().__init__(name,type,HP,attack,defence,dodge_rate)

    def initialize_skill(self):
        return [Aqua_Jet(base_damage = 25),Shield()]

    def __str__(self):
        return "Squirtle"

#小火龙   
class Charmander(FirePokemon):
    def __init__(self,name = "Charmander",type = "Fire",HP = 80,attack = 35,defence = 15,dodge_rate = 10):
        super().__init__(name,type,HP,attack,defence,dodge_rate) 

    def initialize_skill(self):
        return [Ember(base_damage = 35),Flame_Charge(base_damage = 35)]

    def __str__(self):
        return "Charmander"
    



class Play:
    def __init__(self,player_all_pokemon:list,computer_all_pokemon:list):
        self.player_all_pokemon = player_all_pokemon
        self.computer_all_pokemon = computer_all_pokemon
        self.player_team = []
        self.computer_team = []
        self.player_current_pokemon = None
        self.computer_current_pokemon = None

    def player_choose_team(self,player_pokemon_to_choose:list,num = 3):
        print(f"Choose {num} pokemon for your team.")
        while len(self.player_team) < num:
            self.print_pokemon_list(player_pokemon_to_choose)
            choice = input("Choose your pokemon:")
            self.player_team.append(player_pokemon_to_choose[int(choice) - 1])
        print(f"------------------------------------------\nHere is your team:")
        self.print_pokemon_list(self.player_team)

    def computer_choose_team(self, computer_pokemon_to_choose: list, num=3):
            print("\nHere is computer's team:")
            chosen_pokemon = random.sample(computer_pokemon_to_choose, num)
            self.computer_team.extend(chosen_pokemon)
            self.print_pokemon_list(self.computer_team)
            print(f"------------------------------------------")
            time.sleep(1)

    def print_pokemon_list(self,pokemon_list):
        for i,p in enumerate(pokemon_list,1):
            print(f"{i}:{p}")

    def player_choose_pokemon(self):
        print("Your team:")
        self.print_pokemon_list(self.player_team)
        while True:
            choice = input("Choose your pokemon to fight:")
            chosen_pokemon = self.player_team[int(choice) - 1]
            if chosen_pokemon.alive == True:
                print(f"\nYou chose {chosen_pokemon.name}.")
                self.player_current_pokemon = chosen_pokemon
                return chosen_pokemon
            elif chosen_pokemon.alive == False:
                print("This pokemon is fainted!Choose another one.")

    def computer_choose_pokemon(self):
        while True:
            chosen_pokemon = random.choice(self.computer_team)
            if chosen_pokemon.alive == True:
                print(f"Computer chose {chosen_pokemon.name}.\n")
                self.computer_current_pokemon = chosen_pokemon
                return chosen_pokemon
            elif chosen_pokemon.alive == False:
                continue

    def check_alive(self):
        is_player_fail = all(pokemon.alive == False for pokemon in self.player_team)
        is_computer_fail = all(pokemon.alive == False for pokemon in self.computer_team)
        if is_player_fail and is_computer_fail:
            print("All of your pokemon and computer's pokemon is fainted!It is a tie.")
            self.game_finish()
        elif is_player_fail == False and is_computer_fail:
            print("All of computer's pokemon is fainted!You win!")
            self.game_finish()
        elif is_player_fail and is_computer_fail == False:
            print("All of your pokemon is fainted!You lose.")
            self.game_finish()

        if self.player_current_pokemon.alive == False:
            self.player_choose_pokemon()
        time.sleep(1)
        if self.computer_current_pokemon.alive == False:
            self.computer_choose_pokemon()

    def game_finish(self):
        sys.exit()

    def player_choose_skill(self):
        if self.player_current_pokemon.palsy_effect == True:
            return
        print("Your skills:")
        self.player_current_pokemon.skill = self.player_current_pokemon.initialize_skill()
        skills = self.player_current_pokemon.skill
        for i,s in enumerate(skills,1):
            print(f"{i}:{s}")
        choice = input("Choose the skill you want to use :")
        print("\nYour",end = " ")
        self.player_current_skill = self.player_current_pokemon.skill[int(choice) - 1]
        self.player_current_pokemon.apply_skill(self.player_current_skill,self.computer_current_pokemon)
        
    def computer_choose_skill(self):
        if self.computer_current_pokemon.palsy_effect == True:
            return
        self.computer_current_pokemon.skill = self.computer_current_pokemon.initialize_skill()
        skills = self.computer_current_pokemon.skill
        self.computer_current_skill = random.choice(self.computer_current_pokemon.skill)
        print("Computer's",end = " ")
        self.computer_current_pokemon.apply_skill(self.computer_current_skill,self.player_current_pokemon)

#每回合开始前检查存活状态
    def every_round_begin(self):
        self.check_alive()
        self.player_current_pokemon.begin()
        self.computer_current_pokemon.begin()
        self.check_alive()

    def every_round(self):
        self.player_choose_skill()
        time.sleep(2)
        self.computer_choose_skill()
        self.check_alive()

    def run(self):
        self.player_choose_team(self.player_all_pokemon)
        time.sleep(1)
        self.computer_choose_team(self.computer_all_pokemon)
        self.player_current_pokemon = self.player_choose_pokemon()
        self.computer_current_pokemon = self.computer_choose_pokemon()

        while True:
            self.every_round_begin()
            self.every_round()


if __name__ == "__main__":
    pokemon1 = Pikachu()
    pokemon2 = Bulbasaur()
    pokemon3 = Squirtle()
    pokemon4 = Charmander()
    pokemon5 = Pikachu()
    pokemon6 = Bulbasaur()
    pokemon7 = Squirtle()
    pokemon8 = Charmander()

    player_all_pokemon = [pokemon1,pokemon2,pokemon3,pokemon4]
    player_pokemon_to_choose = [pokemon1,pokemon2,pokemon3,pokemon4]
    computer_all_pokemon = [pokemon5,pokemon6,pokemon7,pokemon8]
    computer_pokemon_to_choose = [pokemon5,pokemon6,pokemon7,pokemon8]
    play = Play(player_all_pokemon,computer_all_pokemon)
    play.run()