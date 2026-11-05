from __future__ import annotations
import random
from typing import TYPE_CHECKING
from myself_status import Statuses

if TYPE_CHECKING:
    from myself_pokemon import Pokemon
    from myself_play import Play


class Skills:
    name: str

    def __init__(self, accuracy: float):
        # 初始化技能属性
        self.accuracy = accuracy

    def __str__(self):
       return f'{self.name}'

    def hit(self, user: Pokemon, opponent: Pokemon) -> bool:
        '''判断技能是否命中,命中返回 True'''
        hit_probability = (1 - opponent.evasion) * self.accuracy
        
        if random.random() <= hit_probability:
            return True
        print(f'{self}没能命中...')
        
        # 触发静电
        if opponent.type == 'electric':
            print(f'{opponent.name}触发了【静电】')
            if user.play.is_you(opponent):
                user.play.you_choose_skill()
            else:
                user.play.com_choose_skill()
        
        return False

    def effect(self, user: Pokemon, opponent: Pokemon) -> None:
        '''技能效果'''
        raise NotImplementedError


class damage_skills(Skills):

    def __init__(self, base_damage: float, accuracy: float,
                 effect_chance: float = None,
                 has_secondary_effect: bool = False):
        # 初始化技能属性，并添加新属性
        super().__init__(accuracy)
        self.base_damage = base_damage
        self.has_secondary_effect = has_secondary_effect
        self.effect_chance = effect_chance

    def damage(self, user: Pokemon, opponent: Pokemon) -> int:
        '''计算造成的伤害'''
        effectiveness = user.type_effectiveness(opponent)
        damage = user.attack * self.base_damage * effectiveness - opponent.defense
        damage = max(0,damage)
        return int(damage)

    def secondary_effect_hit(self) -> bool:
        '''判断追加效果是否触发'''
        if random.random() <= self.effect_chance:
            print(f'{self}的追加效果成功触发！！')
            return True
        print(f'{self}的追加效果未能触发...')
        return False

    def secondary_effect(self, user: Pokemon, opponent: Pokemon) -> None:
        '''追加效果'''
        raise NotImplementedError

    def effect(self, user: Pokemon, opponent: Pokemon):
        '''技能效果'''
        if self.hit(user, opponent):
            amount = self.damage(user = user, opponent = opponent)
            user.execute_damage(amount = amount, opponent = opponent)
            if self.has_secondary_effect:
                if self.secondary_effect_hit():
                    self.secondary_effect(user = user, opponent = opponent)


class SeedBomb(damage_skills):
    name = '种子炸弹'
    effect_chance = 0.15

    def __init__(self, base_damage = 0.80, accuracy = 1.00,
                 effect_chance = 0.15,
                 has_secondary_effect = True):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)

    def secondary_effect(self, user, opponent):
        '''追加效果，可能改变对方状态'''
        from myself_status import poison
        poison().apply(target = opponent, duration = -1)
        # -1 表示永久持续该状态


class Thunderbolt(damage_skills):
    name = '十万伏特'

    def __init__(self, base_damage = 1.20, accuracy = 1.00,
                 effect_chance = 0.10,
                 has_secondary_effect = True):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)

    def secondary_effect(self, user, opponent):
        '''追加效果，可能改变对方状态'''
        from myself_status import paralysis
        paralysis().apply(target = opponent, duration = -1)


class QuickAttack(damage_skills):
    name = '电光一闪'

    def __init__(self, base_damage = 1.00,accuracy = 1.00,
                 effect_chance = 0.10,
                 has_secondary_effect = True):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)

    def secondary_effect(self, user, opponent):
        '''追加效果，可能再次释放技能1'''
        user.use_skill(skill = self, opponent = opponent)


class AqueJet(damage_skills):
    name = '水枪'

    def __init__(self, base_damage = 1.40, accuracy = 1.00,
                 effect_chance = None,
                 has_secondary_effect = False):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)


class Ember(damage_skills):
    name = '火花'

    def __init__(self, base_damage = 1.00, accuracy = 1.00,
                 effect_chance = 0.10,
                 has_secondary_effect = True):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)

    def secondary_effect(self, user, opponent):
        '''追加效果，可能改变对方状态'''
        from myself_status import burn
        burn().apply(target = opponent, duration = -1)


class FlameCharge(damage_skills):
    name = '蓄能爆炎'

    def __init__(self, base_damage = 2.00, accuracy = 0.80,
                 effect_chance = 0.80,
                 has_secondary_effect = True):
        super().__init__(base_damage, accuracy,
                         effect_chance, has_secondary_effect)

    def effect(self, user, opponent):
        '''技能效果，直接改变自身和对方的基本属性'''
        from myself_status import after_charge
        after_charge().apply(user, duration = 1)
        print(f'{user}开始蓄力')

    def secondary_effect(self, user, opponent):
        '''追加效果'''
        from myself_status import burn
        burn().apply(target = opponent, duration = -1)

    def effect2(self, user, opponent):
        '''技能效果，直接改变自身和对方的基本属性'''
        if self.hit(user, opponent):
            amount = self.damage(user, opponent)
            user.execute_damage(amount, opponent)
            if self.has_secondary_effect:
                if self.secondary_effect_hit():
                    self.secondary_effect(user = user,
                                          opponent = opponent)
        

class status_skills(Skills):
    pass


class ParasiticSeeds(status_skills):
    name = '寄生种子'

    def __init__(self, accuracy = 1.00):
        super().__init__(accuracy)

    def effect(self, user, opponent):
        '''技能效果，直接改变对方状态'''
        if self.hit(user, opponent):
            from myself_status import parasitic
            parasitic().apply(target = opponent, duration = 3)


class Shield(status_skills):
    name = '护盾'

    def __init__(self, accuracy = 1.00):
        super().__init__(accuracy)

    def effect(self, user, opponent):
        '''技能效果，改变自身状态'''
        from myself_status import damage_reduction
        damage_reduction(amount = 0.50).apply(
            target = user, duration = 1
            )


class Guillotine(status_skills):
    name = '剪刀断头台'

    def __init__(self, accuracy = 0.30):
        super().__init__(accuracy)

    def effect(self, user, opponent):
        '''技能效果，可能改变对手状态'''
        if self.hit(user = user, opponent = opponent):
            print(f'{self}命中了，{opponent}生命降低至0')
            opponent.hp = 0


class Sketch(status_skills):
    name = '写生'

    def __init__(self, accuracy = 1.00):
        super().__init__(accuracy)

    def effect(self, user, opponent):
        '''技能效果，模仿对手技能'''
        mimic = random.choice(opponent.skills)
        print(f'{user.name}模仿了对方的【{mimic.name}】')
        user.use_skill(skill = mimic, opponent = opponent)