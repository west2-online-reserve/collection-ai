import random

import shared

class Buff:
    def __init__(self,target):
        self.name = ""
        self.duration = 0
        self.target = target
    def apply(self):
        pass
    def remove(self):
        pass
    def before_take_damage(self): #受到伤害前触发
        pass
    def after_take_damage(self,status:int,damage:int=0): #受到伤害后触发
        pass
    def after_give_attack(self,status:int,damage:int=0): #攻击后时触发
        pass
    def before_give_attack(self): #攻击前时触发
        pass
    def on_turn_start(self): #回合开始时触发
        pass
    def on_turn_end(self): #回合结束时触发
        self.duration -= 1
        # duration为0时自动删除buff
        if self.duration <= 0:
            self.target.remove_buff(self)


class GrassPokemonBuff(Buff):
    def __init__(self,target):
        super().__init__(target)
        self.name = "GrassPokemonBuff"
        #永久被动，需手动删除
        self.duration = 999
    def on_turn_start(self):
        self.target.heal(self.target.hp*0.1)

class FirePokemonBuff(Buff):
    def __init__(self,target):
        super().__init__(target)
        self.name = "FirePokemonBuff"
        self.duration = 999
        self.amount = 0 #层数
        self.max_amount = 4
        self.attack_add = 0
        self.activated = False
    def apply(self):
        #先移除之前的加成
        self.remove()
        self.attack_add = self.amount * 0.1 * self.target.attack
        self.target.current_attack += self.attack_add
        self.activated = True
    def after_give_attack(self,status:int,damage:int=0):
        if status == shared.damage_success:
            self.amount = min(self.amount+1,self.max_amount)
            #立即激活一次以更新
            self.apply()
    def remove(self):
        self.target.current_attack = self.target.current_attack - self.attack_add
        self.activated = False
    def on_turn_start(self):
        # 回合开始时减少一层
        if self.activated:
            self.remove()
        self.amount = max(self.amount-1,0)
        if self.amount > 0:
            self.apply()

class WaterPokemonBuff(Buff):
    def __init__(self,target):
        super().__init__(target)
        self.name = "WaterPokemonBuff"
        self.duration = 999
    def after_take_damage(self,status:int,damage:int=0):
        if status == shared.damage_success:
            if random.random() < 0.5:
                #减免30%伤害,实际先扣除再回复
                self.target.heal(damage * 0.3)
                print(f'{self.target.name}被动触发! 减免了{damage*0.3}点伤害!')

class ElectricPokemonBuff(Buff):
    def __init__(self,target):
        super().__init__(target)
        self.name = "ElectricPokemonBuff"
        self.duration = 999
    def after_take_damage(self,status:int,damage:int=0):
        if status == shared.damage_evasion:
            print(f'{self.target.name}被动触发! 可立即使用一次技能!')
            if self.target == shared.player_current_pokemon:
                while True:
                    try:
                        index = int(input('输入数字选择技能:'))
                        if index in range(len(shared.player_current_pokemon.skill)):
                            selected_skill = shared.player_current_pokemon.skill[index]
                            break
                    except ValueError:
                        print('输入有误，请重新输入!')
                selected_skill.use(shared.player_current_pokemon, shared.ai_current_pokemon)

                print(
                    f'你的 {shared.player_current_pokemon.name} 当前HP:{shared.player_current_pokemon.current_hp},对方的 {shared.ai_current_pokemon.name} 当前HP:{shared.ai_current_pokemon.current_hp}')
            else:
                random.choice(shared.ai_current_pokemon.skill).use(shared.ai_current_pokemon, shared.player_current_pokemon)
                print(
                    f'你的 {shared.player_current_pokemon.name} 当前HP:{shared.player_current_pokemon.current_hp},对方的 {shared.ai_current_pokemon.name} 当前HP:{shared.ai_current_pokemon.current_hp}')



class Paralysis(Buff):
    def __init__(self,target,duration=1):
        super().__init__(target)
        self.name = "Paralysis"
        self.duration = duration
    def on_turn_start(self):
        self.target.jump_turn = True

    def on_turn_end(self):
        self.target.jump_turn = False
        self.duration -= 1
        # duration为0时自动删除buff
        if self.duration <= 0:
            self.target.remove_buff(self)

class Poison(Buff):
    def __init__(self,target,duration=2):
        super().__init__(target)
        self.name = "Poison"
        self.duration = duration
    def on_turn_start(self):
        self.target.take_damage(self.target.max_hp*0.1,ignore_defense=True,ignore_defense_factor=True,ignore_evasion=True)
        print(f'{self.target.name}中毒! 受到{self.target.max_hp*0.1}点伤害!')

class LifeStealing(Buff):

    def __init__(self,target,enemy,duration=3):
        super().__init__(target)
        self.name = "LifeStealing"
        self.duration = duration
        self.enemy = enemy
        self.bypass_this_turn = False

    def on_turn_start(self):
        if not self.bypass_this_turn:
            amount = self.enemy.hp * 0.1
            self.target.heal(amount)
            self.enemy.take_damage(amount, ignore_defense=True, ignore_defense_factor=True, ignore_evasion=True)
            print(f'{self.target.name}吸取了{self.enemy.name}的生命! 回复了{amount}点血量!')
            self.duration -= 1
            # duration为0时自动删除buff
            if self.duration <= 0:
                self.target.remove_buff(self)
        else:
            self.bypass_this_turn = False

    def apply(self):
        self.bypass_this_turn = True
        amount = self.enemy.hp * 0.1
        self.target.heal(amount)
        self.enemy.take_damage(amount,ignore_defense=True,ignore_defense_factor=True,ignore_evasion=True)
        print(f'{self.target.name}吸取了{self.enemy.name}的生命! 回复了{amount}点血量!')
        self.duration -= 1
        # duration为0时自动删除buff
        if self.duration <= 0:
            self.target.remove_buff(self)

class ShieldSquirtle(Buff):
    def __init__(self,target,duration=1):
        super().__init__(target)
        self.name = "ShieldSquirtle"
        self.duration = duration
    def after_take_damage(self,status:int,damage:int=0):
        if status == shared.damage_success:
            self.target.heal(damage * 0.5)
            print(f'{self.target.name}触发护盾! 减免{damage*0.5}点伤害!')

class Burn(Buff):
    def __init__(self,target,duration=2):
        super().__init__(target)
        self.name = "Burn"
        self.duration = duration
    def on_turn_start(self):
        self.target.take_damage(10,ignore_defense=True,ignore_defense_factor=True,ignore_evasion=True)
        print(f'{self.target.name}受到烧伤! 受到10点伤害!')

