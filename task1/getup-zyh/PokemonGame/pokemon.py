from __future__ import annotations
import random
from typing import List
import skills
from skills import Skill
from effects import Effect
from PIL import Image, ImageTk
import tkinter as tk
import sys



class Pokemon:
    name: str
    type: str
    
    def __init__(self, hp: int, attack: int, defense: int,evasion_rate:float, death_image: str = None) -> None:
        # 初始化 Pokemon 的属性
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.evasion_rate=evasion_rate
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses: List[Effect] = []
        #类型注解，表明statuses是Effect的列表
        self.is_paralyzed = False
        self.is_shielded = False
        self.death_image = death_image

    def initialize_skills(self):
        # 抽象方法，子类应实现具体技能初始化
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon):
        # 使用技能
        print(f"{self.name} uses {skill.name}")
        skill.execute(self, opponent)

    def heal_self(self, amount):
        # 为自身恢复生命值
        if not isinstance(amount, int):
            amount = int(amount)

        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name} heals {amount} HP! Current HP: {self.hp}/{self.max_hp}")

    

    def receive_damage(self, damage:int): 
        if random.random() < self.evasion_rate:
            print(f"{self.name}'s defence absorbed the attack!")
            return
        # 计算伤害并减去防御力，更新 HP
        if hasattr(self, 'is_shielded') and self.is_shielded:
            print(f"{self.name}'s shield reduces the damage by 50%!")
        damage = int(damage * 0.5)  # 减少 50% 的伤害
        
        if not isinstance(damage, int):
            damage = int(damage)

        damage -= self.defense
        if damage <= 0:
            print(f"{self.name}'s defense absorbed the attack!")
            return

        self.hp -= damage
        print(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
            if self.death_image:
                self.show_death_image(self.death_image) #显示死亡图片


    def show_death_image(self, image_path: str):
        # 显示宝可梦死亡图片
        root = tk.Tk()
        root.title(f"{self.name} has fainted!")
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)

        label = tk.Label(root, image=img)
        label.pack()

        #关闭图片后结束程序，即绑定关闭窗口这一事件
        root.protocol("WM_DELETE_WINDOW",lambda: self.end_game(root))
        root.mainloop()

    def end_game(self,root):
        root.quit() #退出图片窗口住循环
        root.destroy() #销毁窗口
        print("Game Over")
        sys.exit() #关闭窗口即可结束，不用强制退出，更优雅


            

    def add_status_effect(self, effect: Effect):
        # 添加状态效果
        self.statuses.append(effect)

    def apply_status_effect(self):
        # 应用所有当前的状态效果，并移除持续时间结束的效果
        for status in self.statuses[:]:  # 使用切片防止列表在遍历时被修改
            status.apply(self)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}'s {status.name} effect has worn off.")
                self.statuses.remove(status)
        #检查是否处于麻痹状态
        if hasattr(self, 'is_paralyzed') and self.is_paralyzed:
            print(f"{self.name} is paralyed and cannot act this turn.")
            return False
        return True
    
    def type_effectiveness(self, opponent: Pokemon):
        # 计算属性克制的抽象方法，具体实现由子类提供
        raise NotImplementedError

    def begin(self):
        # 新回合开始时触发的方法
        pass

    def __str__(self) -> str:
        return f"{self.name} type: {self.type}"


# GrassPokemon 类
class GrassPokemon(Pokemon):
    type = "Grass"

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
        # 每个回合开始时执行玻璃属性特性
        self.grass_attribute()

    def grass_attribute(self):
        # 草属性特性：每回合恢复最大 HP 的 10%
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name} heals {amount} HP at the start of the turn! Current HP: {self.hp}/{self.max_hp}"
        )


# Bulbasaur 类，继承自 GrassPokemon
class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(self, hp=100, attack=50, defense=10, evasion_rate: float = 0.1, death_image: str = None ) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense,evasion_rate, death_image)

    def initialize_skills(self):
        # 初始化技能，具体技能是 SeedBomb 和 ParasiticSeeds
        return [skills.SeedBomb(damage=50), skills.ParasiticSeeds(amount=10)]
    
class ElectricPokemon(Pokemon):
    type = "Electric"

    def __init__(self, hp: int, attack: int, defense: int, evasion_rate: float = 0.1, death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate,death_image)

    def type_effectiveness(self, opponent: Pokemon) -> float:
        # 电属性克制水属性，被草属性克制
        effectiveness = 1.0
        if opponent.type == "Water":
            effectiveness = 2.0  # 电克水，伤害加倍
        elif opponent.type == "Grass":
            effectiveness = 0.5  # 电被草克制，伤害减半
        return effectiveness

    def receive_damage(self, damage: int) -> None:
        if random.random() < self.evasion_rate:
            print(f"{self.name} dodged the attack!")
            # 电属性特性：成功躲闪时，立即使用一次技能
            self.use_skill_immediately()
            return
        # 如果没有躲避，继续正常的伤害计算
        super().receive_damage(damage)

    def use_skill_immediately(self):
        import random
        # 随机选择一个技能并
        if self.skills:
            skill = random.choice(self.skills)  # 随机选择一个技能
            print(f"{self.name} immediately uses {skill.name} after dodging!")
            

class PikaChu(ElectricPokemon):
    name = "PikaChu"

    def __init__(self, hp=80, attack=35, defense=5, evasion_rate=0.3,death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate,death_image)

    def initialize_skills(self):
        return [skills.Thunderbolt(), skills.QuickAttack()]
    
class WaterPokemon(Pokemon):
    type = "Water"

    def __init__(self, hp: int, attack: int, defense: int, evasion_rate: float = 0.1,death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate, death_image)

    def type_effectiveness(self, opponent: Pokemon) -> float:
        # 水属性克制火属性，被电属性克制
        effectiveness = 1.0
        if opponent.type == "Fire":
            effectiveness = 2.0  
        elif opponent.type == "Electric":
            effectiveness = 0.5  
        return effectiveness

    def receive_damage(self, damage: int) -> None:
        import random
        if random.random() < 0.5:
            # 50% 概率减免 30% 的伤害
            reduced_damage = int(damage * 0.7)
            print(f"{self.name} reduced the incoming damage by 30%! Damage taken: {reduced_damage}")
            super().receive_damage(reduced_damage)
        else:
            # 如果没有减免，正常受到伤害
            super().receive_damage(damage)


class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(self, hp=80, attack=25, defense=20, evasion_rate=0.2, death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate,death_image)

    def initialize_skills(self):
        return [skills.AquaJet(), skills.Shield()]
    

class FirePokemon(Pokemon):
    type = "Fire"

    def __init__(self, hp: int, attack: int, defense: int, evasion_rate: float = 0.1, death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate,death_image)
        self.attack_boost = 0  # 攻击力加成，初始为 0 层
        self.max_boost_layers = 4  # 最大叠加层数

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Grass":
            effectiveness = 2.0  
        elif opponent.type == "Water":
            effectiveness = 0.5  
        return effectiveness

    def increase_attack_boost(self) -> None:
        # 每次造成伤害时，叠加 10% 攻击力，最多叠加 4 层
        if self.attack_boost < self.max_boost_layers:
            self.attack_boost += 1
            print(f"{self.name}'s attack is boosted by 10%! Current boost: {self.attack_boost * 10}%")
        else:
            print(f"{self.name}'s attack boost is at the maximum (40%)")

    def get_current_attack(self) -> int:
        # 返回带有加成的当前攻击力
        return int(self.attack * (1 + 0.1 * self.attack_boost))

    def reset_attack_boost(self) -> None:
        # 可以在某些条件下重置攻击加成
        self.attack_boost = 0


class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(self, hp=80, attack=35, defense=15, evasion_rate=0.1, death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate, death_image)

    def initialize_skills(self):
        return [skills.Ember(), skills.FlameCharge()]
    

class WindPokemon(Pokemon):
    type = "Wind"

    def __init__(self, hp: int, attack: int, defense: int, evasion_rate: float = 0.1, death_image: str = None) -> None:
        super().__init__(hp, attack, defense, evasion_rate, death_image)
        self.damage_boost = 0.2  # 躲避后下次技能的伤害提升20%
        self.is_boosted = False  # 是否有加成

    def type_effectiveness(self, opponent: Pokemon) -> float:
        effectiveness = 1.0
        if opponent.type == "Electric":
            effectiveness = 2.0  
        elif opponent.type == "Fire":
            effectiveness = 0.5  
        return effectiveness

    def receive_damage(self, damage: int) -> None:
        if random.random() < self.evasion_rate:
            print(f"{self.name} dodged the attack and gains a damage boost for the next turn!")
            self.is_boosted = True  # 躲避成功后，获得伤害加成
            return
        super().receive_damage(damage)

    def get_current_attack(self) -> int:
        if self.is_boosted:
            print(f"{self.name} uses the boosted attack!")
            attack_value = self.attack * (1 + self.damage_boost)
            self.is_boosted = False  
        else:
            attack_value = self.attack
        return int(attack_value)
    
class Pidgey(WindPokemon):
    name = "Pidgey"

    def __init__(self, hp=60, attack=25, defense=20, evasion_rate=0.15, death_image: str = None) -> None:
            super().__init__(hp, attack, defense, evasion_rate,death_image)

    def initialize_skills(self):
        return [skills.Gust(), skills.AirSlash()]
    

    
    
    
    





    