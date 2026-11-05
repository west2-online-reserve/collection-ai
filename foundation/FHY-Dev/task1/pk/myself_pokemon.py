from __future__ import annotations
import random
import myself_skills
from myself_skills import Skills
from myself_status import Statuses
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from myself_play import Play


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int,
                 attack: int, defense: int,
                 evasion: float) -> None:
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.evasion = evasion
        self.skills: list['Skills'] = self.initialize_skills()
        self.statuses: dict['Statuses', int] = {}
        self.play: Play = None

    def __str__(self) -> str:
        return f'{self.name}'

    def initialize_skills(self) -> list[Skills]:
        '''技能组'''
        raise NotImplementedError

    def type_effectiveness(self, opponent: Pokemon) -> float:
        '''属性克制'''
        raise NotImplementedError

    def use_skill(self, skill: Skills, opponent: Pokemon):
        '''使用技能'''
        # 检查角色状态是否有 cannot_move，并触发
        during_skill_status = [
            i for i in self.statuses.keys()
            if i.is_during_skill and i.name == 'cannot move'
            ]
        if during_skill_status:
            print(f'{self.name}因麻痹而无法行动')
            during_skill_status[0].decrease_duration(self)
            return None
        # 检查角色状态是否有 after_charge，并触发
        during_skill_status = [
            i for i in self.statuses.keys()
            if i.is_during_skill and i.name == 'after charge'
            ]
        if during_skill_status:
            during_skill_status[0].effect(ally = self,
                                          opponent = opponent,
                                          skill = skill)
            during_skill_status[0].decrease_duration(self)
            return None
        
        print(f'{self}使用了【{skill}】')
        skill.effect(user = self, opponent = opponent)

    def execute_damage(self, amount,
                       opponent: Pokemon) -> None:
        '''对对手造成伤害，直接降低对手 hp'''
        # 检查对手是否有伤害减免状态
        during_skill_status = [
            i for i in opponent.statuses.keys()
            if i.is_after_skill
        ]
        if during_skill_status:
            print(f'{opponent}受到的伤害降低')
            status = during_skill_status[0]
            reduce = status.amount
            amount *=(1 - reduce)
            amount = int(amount)
            status.decrease_duration(ally = opponent)

        opponent.hp -= amount
        opponent.hp = max(opponent.hp, 0)
        print(f'其受到{amount}点伤害，{opponent}生命降低至{opponent.hp}')

    def heal_self(self, amount: int) -> None:
        '''对自身进行治疗，直接增加自身 hp'''
        self.hp += amount
        self.hp = min(self.max_hp, self.hp)
        print(f'{self}回复了{amount}点生命，生命恢复至{self.hp}')

    def after_round(self, opponent: Pokemon) -> None:
        '''回合结束后触发'''
        for i in self.statuses.copy().keys():
            if i.is_after_round:
                i.effect(ally = self, opponent = opponent)
                i.decrease_duration(ally = self)

    def before_skill(self, opponent: Pokemon):
        '''技能释放前触发'''
        for i in self.statuses.copy().keys():
            if i.is_before_skill:
                i.effect(ally = self, opponent = opponent)
                i.decrease_duration(ally = self)


class Grass_Pokemon(Pokemon):
    type = 'grass'
    type_name = '草'

    def grass_attribute(self):
        '''草系角色的被动：恢复生命,直接改变自身hp'''
        print(f'{self}触发了【茂盛】')
        amount = self.max_hp * 0.1
        self.heal_self(amount = amount)

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        if opponent.type == 'water':
            effectiveness = 1.5
        if opponent.type == 'fire':
            effectiveness = 0.75
        return effectiveness

    def after_round(self, opponent):
        super().after_round(opponent)
        self.grass_attribute()


class Bulbasaur(Grass_Pokemon):
    name = '妙蛙种子'

    def __init__(self, hp = 200, attack = 25,
                 defense = 20, evasion = 0.10):
        super().__init__(hp, attack, defense, evasion)

    def initialize_skills(self):
        return [myself_skills.SeedBomb(),
                myself_skills.ParasiticSeeds()]


class Electric_Pokemon(Pokemon):
    type = 'electric'
    type_name = '电'
    # electric_attribute 请移步 Skills.hit()

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        if opponent.type == 'water':
            effectiveness = 1.5
        if opponent.type == 'grass':
            effectiveness = 0.75
        return effectiveness


class PikaChu(Electric_Pokemon):
    name = '皮卡丘'

    def __init__(self, hp = 160, attack = 35,
                 defense = 10, evasion = 0.30):
        super().__init__(hp, attack, defense, evasion)

    def initialize_skills(self):
        return [myself_skills.Thunderbolt(),
                myself_skills.QuickAttack()]


class Water_Pokemon(Pokemon):
    type = 'water'
    type_name = '水'
    
    def water_attribute(self):
        if random.random() <= 0.50:
            from myself_status import damage_reduction
            damage_reduction(amount = 0.30).apply(
                target = self, duration = 1
                )
            print(f'{self.name}在本回合获得了【激流】')

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        if opponent.type == 'fire':
            effectiveness = 1.5
        if opponent.type == 'grass' or opponent.type == 'electric':
            effectiveness = 0.75
        return effectiveness

    def before_skill(self, opponent):
        super().before_skill(opponent)
        self.water_attribute()


class Squirtle(Water_Pokemon):
    name = '杰尼龟'

    def __init__(self, hp = 160, attack = 35,
                 defense = 20, evasion = 0.20):
        super().__init__(hp, attack, defense, evasion)

    def initialize_skills(self):
        return [myself_skills.AqueJet(),
                myself_skills.Shield()]


class Fire_Pokemon(Pokemon):
    type = 'fire'
    type_name = '火'

    def __init__(self, hp, attack, defense, evasion):
        super().__init__(hp, attack, defense, evasion)
        self.original_attack = attack

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        if opponent.type == 'grass':
            effectiveness = 1.5
        if opponent.type == 'water':
            effectiveness = 0.75
        return effectiveness

    def execute_damage(self, amount,
                       opponent: Pokemon) -> None:
        super().execute_damage(amount, opponent)
        # 火属性特性，造成攻击时攻击力提高
        if self.attack <= self.original_attack * 1.4:
            print(f'{self.name}触发【猛火】的攻击力提高')
            self.attack += self.original_attack * 0.1

class Charmander(Fire_Pokemon):
    name = '小火龙'

    def __init__(self, hp = 160, attack = 40,
                 defense = 15, evasion = 0.10):
        super().__init__(hp, attack, defense, evasion)

    def initialize_skills(self):
        return [myself_skills.Ember(),
                myself_skills.FlameCharge()]


class Normal_Pokemon(Pokemon):
    type = 'normal'
    type_name = '一般'

    def __init__(self, hp, attack, defense, evasion):
        super().__init__(hp, attack, defense, evasion)
        self.original_attack = attack

    def type_effectiveness(self, opponent):
        effectiveness = 1.0
        return effectiveness

    def normal_attribute(self):
        print(f'{self.name}触发了【心情不定】')
        up = random.random()
        down = random.random()
        if up >=0.66:
            self.attack += self.original_attack*0.1
            print('攻击力提高')
        elif up>=0.33:
            self.defense += 10
            print('防御力提高')
        else:
            self.evasion +=0.10
            print('闪避率提高')
        if down >=0.66:
            self.attack -= self.original_attack*0.1
            print('攻击力降低')
        elif down>=0.33:
            self.defense -= 10
            print('防御力降低')
        else:
            self.evasion -=0.10
            print('闪避率降低')
        


class Smeargle(Normal_Pokemon):
    name = '图图犬'

    def __init__(self, hp = 160, attack = 30,
                 defense = 15, evasion = 0.15):
        super().__init__(hp, attack, defense, evasion)

    def initialize_skills(self):
        return [myself_skills.Guillotine(),
                myself_skills.Sketch()]
