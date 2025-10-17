'''这个文档来定义宝可梦类'''
import skill,random
d = {'属性': '被克制', '草': '火', '火': '水', '水': '电', '电': '草', '毒': '水','无':'无'}
d1 = {'属性': '克制', '草': '电', '火': '草', '水': '火', '电': '水', '毒': '草','无':'无'}
class Pokemon(object):  
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, attribute: str,skills:list) -> None:  
        self.name = name  
        self.hp = hp  
        self.max_hp = hp  
        self.attack = attack  
        self.defense = defense  
        self.dodge_chance = dodge_chance  
        self.attribute = attribute  
        self.skills = skills
        self.status_effects = {}
        self.extra_skill = False
        self.dodge_attack = False

    def choose_skill(self):
        print(f"\n请选择 {self.name} 的技能:")
        for i, skill in enumerate(self.skills):
            print(f"{i + 1}. {skill.name}")
        choice = int(input("输入数字选择技能: "))
        return self.skills[choice - 1]
    
    def cpu_choose_skill(self):
        choice = random.randint(1,2)
        return self.skills[choice - 1]
    
    def display_information(self):
        print(f"宝可梦: {self.name}")
        print(f"生命值: {self.max_hp}")
        print(f"攻击力: {self.attack}")
        print(f"防御力: {self.defense}")
        print(f"闪避几率: {self.dodge_chance}%")
        print(f"属性: {self.attribute}")
        print(f"技能:")
        for skill in self.skills:
            print(f"- {skill.name}: {skill.effect_description}")
    def apply_status_effect(self, effect: str, duration: int):
        self.status_effects[effect] = duration

    def dodge(self) -> bool:
        """检查是否成功闪避攻击"""
        if random.random() < self.dodge_chance / 100:
            print(f'{self.name}成功闪避! 当前血量为 {self.hp}.')
            self.dodge_attack = True
            return True
        return False

    def adjust_damage(self, damage: int, attacking_attribute: str) -> int:
        """根据防御力和属性调整伤害"""

        if attacking_attribute == d.get(self.attribute, ''):  # 被克制
            print(f'{self.name}受到{d[self.attribute]}属性克制, 伤害由原来的{damage}翻倍, 变为{damage * 2}')
            damage *= 2
        elif attacking_attribute == d1.get(self.attribute, ''):  # 克制对方
            print(f'{self.name}的属性为{self.attribute}克制{d1[self.attribute]}属性, 伤害由原来的{damage}减半, 变为{damage // 2}')
            damage //= 2
        damage = max(0, damage - self.defense)  # 减去防御值，伤害不能为负数
        print(f'根据{self.name}的防御力{self.defense},伤害降低为{damage}')
        return damage
    
    def apply_damage(self, damage: int) -> None:
        """进行伤害计算并更新血量"""
        if self.hp - damage>=0:
            self.hp -= damage
            print(f'{self.name}受到了{damage}点伤害! 现在的血量为{self.hp}.')
        else:
            self.hp = 0
            print(f'{self.name}受到{damage}点伤害,血量下降至0以下，即将昏厥.')

    def take_damage(self, damage: int, attacking_attribute: str) -> None:
        """总控制函数，负责如何处理受伤"""
        if self.dodge():

            return  # 如果成功闪避，直接返回

        damage = self.adjust_damage(damage, attacking_attribute)  # 调整伤害

        # 检查护盾状态
        if '减伤' in self.status_effects:
            damage //= 2
            print(f'{self.name} 的护盾使伤害减半，伤害变为 {damage}')
            self.status_effects['减伤'] -= 1
            if self.status_effects['减伤'] <= 0:
                del self.status_effects['减伤']

        self.apply_damage(damage)

    def is_fainted(self) -> bool:
        """检查是否昏厥"""   
        if self.hp <= 0:  
            return True  
        return False  

# 定义火属性的 Pokemon
class FirePokemon(Pokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, '火', skills)

    def passive_skills(self):
        print(f'根据火属性的被动，{self.name}的攻击力上升了')
        add = self.attack//5
        self.attack += add 

class Charmander(FirePokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, skills)

# 定义水属性的 Pokemon
class WaterPokemon(Pokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, '水', skills)

    def passive_skills(self):
        pass

    def apply_damage(self, damage: int) -> None:
        if random.random() < 0.5:
            print(f'{self.name}触发了水属性被动，伤害减免了30%, 由{damage}变为{int(damage * 0.7)}')
            damage = int(damage * 0.7)
        super().apply_damage(damage)

class Squirtle(WaterPokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, skills)

# 定义草属性的 Pokemon
class GrassPokemon(Pokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, '草', skills)

    def passive_skills(self):
        recover = int(self.max_hp / 10)
        if self.hp + recover<=self.max_hp:
            self.hp += recover
            print(f'根据草属性的被动技能, {self.name}的生命值回复了{recover}, 现在的生命值为{self.hp}')
        else:
            self.hp = self.max_hp
            print(f'由于草属性被动，{self.name}的生命值恢复至满血,现在的生命值为{self.max_hp}')

class Bulbasaur(GrassPokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, skills)

# 定义电属性的 Pokemon
class ElectricPokemon(Pokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, '电', skills)

    def passive_skills(self):
        pass

    def dodge(self) -> bool:
        if super().dodge():
            print('由于电属性被动，可以立即使用一次技能')
            self.extra_skill = True
            return True
        return False

class Pikachu(ElectricPokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, skills)

# 定义无属性的 Pokemon
class NoattributePokemon(Pokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, '无', skills)
        self.transform = False

    def passive_skills(self):
        if self.transform == True:
            pass

class Tianfa(NoattributePokemon):
    def __init__(self, name: str, hp: int, attack: int, defense: int, dodge_chance: int, skills: list) -> None:
        super().__init__(name, hp, attack, defense, dodge_chance, skills)

class Pokemon_team(object):
    def __init__(self) -> None:
        self.pokemons = []
    
    def clear(self):
        self.pokemons.clear()

    def remove_pokemon(self, pokemon: Pokemon):
        if pokemon in self.pokemons:
            self.pokemons.remove(pokemon)

    def __len__(self):
        return len(self.pokemons)

    def append(self,pokemon:Pokemon):
        self.pokemons.append(pokemon)
    
    def display(self):
        for i,v in enumerate(self.pokemons):
            print(f'{i+1}.{v.name}',end = ' ')
    
    def cup_choose_pokemon(self):
      choice = random.randint(0,len(self.pokemons)-1)
      print(f'\n电脑选择了{self.pokemons[choice].name}来出战')
      return self.pokemons[choice]

    def choose_pokemon(self):
      print('\n请选择要出战的宝可梦:')
      self.display()
      out = int(input('\n输入数字选择你的宝可梦: '))
      print(f'你选择了{self.pokemons[out-1].name}来出战')
      return self.pokemons[out-1]

    def introduction(self):
        for i,v in enumerate(self.pokemons):
            print(f'{i+1}.{v.name}',end = ' ')
        correct_list = list(map(str,range(1,i+2)))
        print('\n请选择你想查看的宝可梦:')
        while True:
            choice = input()
            if len(choice)==1 and choice in correct_list:
                choice = int(choice)
                break
            else:
                print('请输入有效的数字:')
        self.pokemons[choice-1].display_information()


pikachu = Pikachu('皮卡丘', 80, 35, 5, 30, [skill.thunderbolt, skill.quick_attack])
bulbasaur = Bulbasaur('妙蛙种子', 100, 35, 10, 10, [skill.seed_bomb, skill.parasitic_seeds])
squirtle = Squirtle('杰尼龟', 80, 25, 20, 20, [skill.water_gun, skill.shield])
charmander = Charmander('小火龙', 80, 35, 15, 10, [skill.ember, skill.flame_charge])
tianfa = Tianfa('天罚',100,30,10,15,[skill.ordinary_rooster,skill.mimicry])

all_pokemons = Pokemon_team()
player_team = Pokemon_team()
computer_team = Pokemon_team()
all_pokemons.append(pikachu)
all_pokemons.append(bulbasaur)
all_pokemons.append(squirtle)
all_pokemons.append(charmander)
all_pokemons.append(tianfa)
if __name__ == '__main__':
    all_pokemons.introduction()


