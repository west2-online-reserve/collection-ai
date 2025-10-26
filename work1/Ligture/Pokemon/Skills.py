import random
import Buff
import shared

class Skill:
    def __init__(self, name):
        self.name = name
    def use(self,source_pokemon,target_pokemon):
        pass
    def print_info(self,source_pokemon,target_pokemon,status,damage):
        if status == shared.damage_success:
            print(f" ({source_pokemon.belong}){source_pokemon.name} 使用了 {self.name} !对 ({target_pokemon.belong}){target_pokemon.name} 造成了 {damage} 点伤害!")
        elif status == shared.damage_evasion:
            print(f" ({source_pokemon.belong}){source_pokemon.name} 使用了 {self.name} !但 ({target_pokemon.belong}){target_pokemon.name} 灵巧地躲过了...")
        else:
            print(f" ({source_pokemon.belong}){source_pokemon.name} 使用了 {self.name} !但被格挡了,成功造成了0点伤害!")

class Thunderbolt(Skill):
    def __init__(self):
        super().__init__("十万伏特")

    def use(self,source_pokemon,target_pokemon):
        status,damage = source_pokemon.give_attack(target_pokemon,1.4*source_pokemon.current_attack*source_pokemon.attack_factor,type = shared.attr_electric)
        self.print_info(source_pokemon,target_pokemon,status,damage)
        if random.random() < 0.1:
            target_pokemon.add_buff(Buff.Paralysis(target_pokemon,1))
            print(f'({source_pokemon.belong}){source_pokemon.name}使用了十万伏特!对({target_pokemon.belong}){target_pokemon.name}造成麻痹!(1回合)')

class QuickAttack(Skill):
    def __init__(self):
        super().__init__("电光一闪")

    def use(self,source_pokemon,target_pokemon):
        status,damage = source_pokemon.give_attack(target_pokemon,1*source_pokemon.current_attack*source_pokemon.attack_factor)
        self.print_info(source_pokemon, target_pokemon, status, damage)
        if random.random() < 0.1:
            print(f'({source_pokemon.belong}){source_pokemon.name}触发电光一闪的效果!获得一次额外行动机会!')
            status, damage = source_pokemon.give_attack(target_pokemon,1 * source_pokemon.current_attack * source_pokemon.attack_factor)
            self.print_info(source_pokemon, target_pokemon, status, damage)

class SeedBomb(Skill):
    def __init__(self):
        super().__init__("种子炸弹")

    def use(self,source_pokemon,target_pokemon):
        status,damage = source_pokemon.give_attack(target_pokemon,1*source_pokemon.current_attack*source_pokemon.attack_factor,type = shared.attr_grass)
        self.print_info(source_pokemon,target_pokemon,status,damage)
        if status == shared.damage_success:
            if random.random() < 0.15:
                target_pokemon.add_buff(Buff.GrassPokemonBuff(target_pokemon))
                print(f'({source_pokemon.belong}){source_pokemon.name}使用了种子炸弹!对({target_pokemon.belong}){target_pokemon.name}造成了中毒效果(2回合)!')

class ParasiticSeed(Skill):
    def __init__(self):
        super().__init__("寄生种子")
    def use(self,source_pokemon,target_pokemon):
        buff = Buff.LifeStealing(source_pokemon,target_pokemon)
        source_pokemon.add_buff(buff)
        print(f'({source_pokemon.belong}){source_pokemon.name}使用了寄生种子!每回合开始时吸取({target_pokemon.belong}){target_pokemon.name}10%的血量!(3回合)')
        #手动触发一次
        buff.apply()

class AquaJet(Skill):
    def __init__(self):
        super().__init__("水枪")

    def use(self,source_pokemon,target_pokemon):
        status,damage = source_pokemon.give_attack(target_pokemon,1.4*source_pokemon.current_attack*source_pokemon.attack_factor,type = shared.attr_water)
        self.print_info(source_pokemon,target_pokemon,status,damage)

class ShieldSquirtle(Skill):
    def __init__(self):
        super().__init__("护盾")

    def use(self,source_pokemon,target_pokemon):
        #通过buff实现护盾效果
        buff = Buff.ShieldSquirtle(source_pokemon)
        source_pokemon.add_buff(buff)
        print(f'({source_pokemon.belong}){source_pokemon.name}使用了护盾!减少下一回合受到伤害的50%!')

class Ember(Skill):
    def __init__(self):
        super().__init__("火花")

    def use(self,source_pokemon,target_pokemon):
        status,damage = source_pokemon.give_attack(target_pokemon,1*source_pokemon.current_attack*source_pokemon.attack_factor,type = shared.attr_fire)
        self.print_info(source_pokemon,target_pokemon,status,damage)
        if random.random() < 0.1 and status == shared.damage_success:
            target_pokemon.add_buff(Buff.Burn(target_pokemon, 2))
            print(f'({source_pokemon.belong}){source_pokemon.name}使用了火花!对({target_pokemon.belong}){target_pokemon.name}造成了烧伤效果(2回合)!')

class FlameCharge(Skill):
    def __init__(self):
        super().__init__("蓄能爆炎")
        self.cd = False

    def use(self,source_pokemon,target_pokemon):
        #临时增加15闪避率
        target_pokemon.current_evasion = target_pokemon.evasion + 0.15
        status, damage = source_pokemon.give_attack(target_pokemon,3 * source_pokemon.current_attack * source_pokemon.attack_factor,type=shared.attr_fire)
        if status == shared.damage_success and random.random() < 0.8:
            target_pokemon.add_buff(Buff.Burn(target_pokemon, 2))
            print(f'({source_pokemon.belong}){source_pokemon.name}使用了蓄能爆炎!对({target_pokemon.belong}){target_pokemon.name}造成了烧伤效果(2回合)!')
        self.print_info(source_pokemon, target_pokemon, status, damage)
        #恢复闪避率
        target_pokemon.current_evasion -= 0.15


class Roost(Skill):
    def __init__(self):
        super().__init__("羽栖")

    def use(self,source_pokemon,target_pokemon):
        heal_amount = source_pokemon.hp * 0.5
        source_pokemon.heal(heal_amount)
        print(f'({source_pokemon.belong}){source_pokemon.name}使用了羽栖!回复了{heal_amount}点血量!')