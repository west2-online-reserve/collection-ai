from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
from play import *
import random


class Pokemon:
    name: str
    type: str

    def __init__(self, hp: int, attack: int, defense: int,avoid: int) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.avoid = avoid
        self.hit_rate = 100
        self.aviod_state =False
        self.operator = None
        self.skills = self.initialize_skills()
        self.delay_skill = None
        self.alive = True
        self.statuses = []
        self.shield = False
        self.round = 0

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError
    
    def avoid_determind(self,opponent:Pokemon):
        if random.randint(1,100) >= opponent.avoid and random.randint(1,100) <= self.hit_rate:
            return False
        else:
            opponent.aviod_state = True
            return True

        # 使用技能
    def use_skill(self, skill: Skill, opponent: Pokemon,play: Play):
        
        if skill.skill_type == 'delay' and self.delay_skill == None:
            print(f"{self.operator}的 {self.name} 使用了 {skill.name}      延迟技能")
            skill.execute(self, opponent)
            return
        elif self.avoid_determind(opponent):  
            print(f"{self.operator}的技能未命中")
            opponent.passive_attack(play)
            #延迟攻击被闪避后，重置self.delay_skill
            self.delay_skill = None
            
        else:
            print(f"{self.operator}的 {self.name} 使用了 {skill.name}      普通技能")
            skill.execute(self, opponent)
            self.delay_skill = None
            

        
    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} 恢复了 {amount} 点生命值! 当前生命值: {self.hp}/{self.max_hp}")

    def receive_true_damege(self,damage):
        if not isinstance(damage, int):
            damage = int(damage)
        self.hp -= damage
        print(
            f"{self.operator}的 {self.name} 受到{damage}点伤害！剩余生命值：{self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 战败！")

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.operator}的 {self.name} 的防御完全抵挡了攻击!")
            return

        self.hp -= damage
        print(
            f"{self.operator}的 {self.name} 受到{damage}点伤害！剩余生命值：{self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 战败!")

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.operator} {self.name} 的 {status.name} 效果已经消失。")
                self.statuses.remove(status)

    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        self.round += 1
        # 新回合开始时触发的方法
        

    def passive_attack(self,play):
        pass

    def skip_use_skill(self,skill = None):
        whether_skip = False
        return [whether_skip,skill]
    
    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"
    



# GlassPokemon 类
class GlassPokemon(Pokemon):
    type = "Glass"

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Water":
            effectiveness = 2.0
        elif opponent_type == "Fire":
            effectiveness = 0.5
        return effectiveness

    def begin(self):
        super().begin()
        # 每个回合开始时执行玻璃属性特性
        self.glass_attribute()

    def glass_attribute(self):
        # 玻璃属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} 在回合开始时恢复了 {amount} 点生命值! 当前生命值: {self.hp}/{self.max_hp}"
        )


# Bulbasaur 类，继承自 GlassPokemon
class Bulbasaur(GlassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=200, attack=50, defense=10,avoid=10) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense,avoid)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]


class ThunderPokemon(Pokemon):
    type = 'Thunder'

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Water":
            effectiveness = 2.0
        elif opponent_type == "Glass":
            effectiveness = 0.5
        return effectiveness
    def passive_attack(self,play):
        self.thunder_attribute(play)
    
    def thunder_attribute(self,play):
        if self.aviod_state:
            print('是时候反击了')
            if self.operator =='player':
                play.player_use_skills()
            if self.operator =='computer':
                play.computer_use_skills()
            self.avoid_state = False

    
        
        
class PikaChu(ThunderPokemon):
    name = 'Pikachu'

    def __init__(self,hp=160,attack=35,defense=5,avoid=30):
        super().__init__(hp,attack,defense,avoid)
    
    def initialize_skills(self):
        return [skills.Thunderbolt(),skills.QuickAttack()]
    
class Waterpokemon(Pokemon):
    type = 'Water'

    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Fire":
            effectiveness = 2.0
        elif opponent_type == "Thunder":
            effectiveness = 0.5
        return effectiveness
    
    #天赋：受到伤害时，有50%的几率减免30%的伤害 
    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.operator}的 {self.name} 的防御完全抵挡了攻击!")
            return
        if random.randint(1,100) <= 50:
            print('伤害减半')
        

        self.hp -= damage
        print(
            f"{self.operator}的 {self.name} 受到了 {damage} 点伤害! 剩余生命值: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 战败了!")

class Squirtle(Waterpokemon):
    name = 'Squirtle'

    def __init__(self,hp=160,attack=25,defense=20,avoid=20):
        super().__init__(hp,attack,defense,avoid)
        
    def initialize_skills(self):
        return [skills.AquaJet(),skills.Shield()]
    

    def receive_damage(self, damage):
        # 计算伤害并减去防御力，更新 HP
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.operator}的 {self.name} 的防御完全抵挡了攻击!")
            return
        if random.randint(1,100) <= 50:
            damage *= 0.5
            print('伤害减半')

        if self.shield:
            damage *= 0.5
            print('护盾生效')
            self.shield = False

        self.hp -= damage
        print(
            f"{self.operator}的 {self.name} 受到了 {damage} 点伤害! 剩余生命值: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 战败了!")
    
class Firepokemon(Pokemon):
    type = 'Fire'

    def __init__(self,hp,attack,defense,avoid):
        super().__init__(hp,attack,defense,avoid)
        self.attack_increase_layers = 0
        self.max_attack_increase_layers = 4
    
    def type_effectiveness(self, opponent: Pokemon):
        # 针对敌方 Pokemon 的类型，调整效果倍率
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "Glass":
            effectiveness = 2.0
        elif opponent_type == "Water":
            effectiveness = 0.5
        return effectiveness
    #火系天赋：每次造成伤害，叠加10%攻击力，最多4层
    def use_skill(self, skill: Skill, opponent: Pokemon,play: Play):
        if skill.skill_type == 'delay' and self.delay_skill == None:
            print(f"{self.operator}的 {self.name} 使用了 {skill.name}      延迟技能")
            skill.execute(self, opponent)
            return
        
        elif self.avoid_determind(opponent):  
            print(f"{self.operator}的技能未命中")
            opponent.passive_attack(play)
            self.delay_skill = None

        else:
            print(f"{self.operator}的 {self.name} 使用了 {skill.name}      普通技能")
            skill.execute(self, opponent)
            self.delay_skill = None
            if self.attack_increase_layers < self.max_attack_increase_layers:
                self.attack_increase_layers += 1
                self.attack *= 1.1

class Charmander(Firepokemon):
    name = "Charmander"

    def __init__(self,hp=160,attack=25,defense=20,avoid=20):
        super().__init__(hp,attack,defense,avoid)
        
       
        self.FlameCharge_used_round = 0

    def initialize_skills(self):
        return [skills.Ember(),skills.FlameCharge()]