import Buff
import random
import shared
import Skills


class Pokemon:
    def __init__(self,hp,attack,evasion,defense,belong = 'player'):
        self.name = ''
        self.hp = hp
        self.current_hp = hp
        self.attack = attack
        self.current_attack = attack
        self.evasion = evasion
        self.current_evasion = evasion
        self.defense = defense
        self.current_defense = defense
        #伤害倍率
        self.attack_factor = 1.0
        self.attribute = 'normal'
        #实际受伤倍率
        self.damage_factor = 1.0
        self.counter = 2 #0:被克制 1:克制 2:无克制
        self.buff:list[Buff.Buff] = []
        self.skill = []
        self.jump_turn = False
        self.belong = belong
    def activate_counter(self):
        #攻击倍率翻倍
        self.attack_factor *= 2
        self.counter = 1
    def activate_be_countered(self):
        #攻击倍率减半
        self.attack_factor /= 2
        self.counter = 0
    def deactivate_counter(self):
        #恢复攻击倍率
        if self.counter == 1:
            self.attack_factor /= 2
            self.counter = 2
        elif self.counter == 0:
            self.attack_factor *= 2
            self.counter = 2
    def add_buff(self, buff: Buff):
        self.buff.append(buff)
    def remove_buff(self, buff: Buff):
        self.buff.remove(buff)
    def take_damage(self, damage: int, ignore_evasion: bool = False, ignore_defense: bool = False, ignore_defense_factor: bool = False) -> (int,float):
        #返回值(状态,实际伤害) 0闪避 1命中 2格挡(强制扣除1点血量)
        # 先判断是否闪避
        for buff in self.buff:
            buff.before_take_damage()
        is_evasion_success = True if random.random() <= self.current_evasion else False
        if not ignore_evasion:
            if is_evasion_success:
                for buff in self.buff:
                    buff.after_take_damage(shared.damage_evasion,0)
                return shared.damage_evasion,0
        if not ignore_defense:
            damage = damage - self.current_defense
        if not ignore_defense_factor:
            damage = damage * self.damage_factor
        if damage > 1:
            self.current_hp -= damage
            for buff in self.buff:
                buff.after_take_damage(shared.damage_success, damage)
            return shared.damage_success,damage
        else:
            for buff in self.buff:
                buff.after_take_damage(shared.damage_block, 0)
            return shared.damage_block,0
    def heal(self,amount):
        #回复血量，不能超过最大血量
        if amount + self.current_hp <= self.hp:
            self.current_hp += amount
        else:
            self.current_hp = self.hp
    def turn_start(self):
        for buff in self.buff:
            buff.on_turn_start()
    def turn_end(self):
        for buff in self.buff:
            buff.on_turn_end()
    def give_attack(self,target:'Pokemon',damage:int,ignore_evasion: bool = False, ignore_defense: bool = False, ignore_defense_factor: bool = False,type=shared.attr_normal) -> (int,float):
        for buff in self.buff:
            buff.before_give_attack()
        status,real_damage = target.take_damage(damage,ignore_evasion,ignore_defense,ignore_defense_factor)
        for buff in self.buff:
            buff.after_give_attack(status,real_damage)
        return status,real_damage
    def __str__(self):
        return f'{self.name}(HP:{self.current_hp}/{self.hp} ATK:{self.current_attack}/{self.attack} DEF:{self.current_defense}/{self.defense} EVA:{self.current_evasion*100}%/{self.evasion*100}%),atk_factor:{self.attack_factor},def_factor:{self.damage_factor},attr:{self.attribute},counter:{self.counter}'

class GrassPokemon(Pokemon):
    def __init__(self,hp,attack,evasion,defense):
        super().__init__(hp,attack,evasion,defense)
        self.attribute = shared.attr_grass
        #添加属性buff
        grass_buff = Buff.GrassPokemonBuff(self)
        self.add_buff(grass_buff)

class FirePokemon(Pokemon):
    def __init__(self,hp,attack,evasion,defense):
        super().__init__(hp,attack,evasion,defense)
        self.attribute = shared.attr_fire
        #添加属性buff
        fire_buff = Buff.FirePokemonBuff(self)
        self.add_buff(fire_buff)

class WaterPokemon(Pokemon):
    def __init__(self,hp,attack,evasion,defense):
        super().__init__(hp,attack,evasion,defense)
        self.attribute = shared.attr_water
        #添加属性buff
        water_buff = Buff.WaterPokemonBuff(self)
        self.add_buff(water_buff)

class ElectricPokemon(Pokemon):
    def __init__(self,hp,attack,evasion,defense):
        super().__init__(hp,attack,evasion,defense)
        self.attribute = shared.attr_electric
        #添加属性buff
        electric_buff = Buff.ElectricPokemonBuff(self)
        self.add_buff(electric_buff)

class PikaChu(ElectricPokemon):
    def __init__(self):
        super().__init__(hp=80,attack=35,evasion=0.3,defense=5)
        self.name = "皮卡丘"
        self.skill = [Skills.Thunderbolt(),Skills.QuickAttack()]

class Bulbasaur(GrassPokemon):
    def __init__(self):
        super().__init__(hp=100,attack=35,evasion=0.1,defense=10)
        self.name = "妙蛙种子"
        self.skill = [Skills.SeedBomb(),Skills.ParasiticSeed()]

class Squirtle(WaterPokemon):
    def __init__(self):
        super().__init__(hp=80,attack=25,evasion=0.2,defense=20)
        self.name = "杰尼龟"
        self.skill = [Skills.AquaJet(),Skills.ShieldSquirtle()]

class Charmander(FirePokemon):
    def __init__(self):
        super().__init__(hp=80,attack=35,evasion=0.1,defense=15)
        self.name = "小火龙"
        self.flame_charge = Skills.FlameCharge()
        self.skill = [Skills.Ember(),self.flame_charge]
        self.cd = False
    def turn_start(self):
        super().turn_start()
        #cd只有一回合，直接取反
        self.cd = not self.cd
        if self.cd:
            self.skill.remove(self.flame_charge)
        else:
            self.skill.append(self.flame_charge)


class Pidgey(Pokemon):
    def __init__(self):
        super().__init__(hp=70,attack=25,evasion=0.5,defense=5)
        self.name = "波波"
        self.skill = [Skills.Roost(),Skills.QuickAttack()]