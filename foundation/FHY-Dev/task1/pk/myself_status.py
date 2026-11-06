from __future__ import annotations
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from myself_pokemon import Pokemon
    from myself_skills import Skills

class Statuses:
    name: str

    def __init__(self, is_after_round: bool,
                 is_before_skill: bool,
                 is_during_skill: bool,
                 is_after_skill: bool):
        self.is_after_round = is_after_round
        self.is_before_skill = is_before_skill
        self.is_during_skill = is_during_skill
        self.is_after_skill = is_after_skill

    def __str__(self):
        return f'{self.name}'

    def apply(self, target: Pokemon, duration: int) -> None:
        '''赋予状态'''
        has_this = False
        for i in target.statuses.keys():
               if i.name == self.name:
                   has_this = True
                   break

        if not has_this:
            target.statuses[self] = duration
            print(f'{target}{self}了')

    def decrease_duration(self, ally: Pokemon) -> None:
        '''减少状态持续时间'''
        duration = ally.statuses[self]
        duration -= 1
        ally.statuses[self] = duration

        if duration == 0:
            del ally.statuses[self]
            if not self.is_during_skill and not self.is_after_skill:
                print(f'{ally}的{self}状态解除了')

    def effect(self, ally: Pokemon, opponent: Pokemon) -> None:
        '''状态触发时的效果'''
        raise NotImplementedError


class after_round_statuses(Statuses):
    # 指在回合后触发的状态
    def __init__(self, is_after_round = True,
                 is_before_skill = False,
                 is_during_skill = False,
                 is_after_skill = False):
        super().__init__(is_after_round,
                         is_before_skill,
                         is_during_skill,
                         is_after_skill)


class poison(after_round_statuses):
    name = '中毒'

    def effect(self, ally, opponent):
        '''中毒状态触发时的效果，直接改变目标的hp'''
        amount = ally.max_hp * 0.1
        print(f'{ally.name}因{self.name}的影响')
        opponent.execute_damage(amount = amount, opponent = ally)


class parasitic(after_round_statuses):
    name = '寄生'

    def effect(self, ally, opponent):
        '''寄生状态触发时的效果，直接改变自身和对方的hp'''
        amount = ally.max_hp * 0.1
        print(f'{ally.name}因{self.name}的影响')
        opponent.execute_damage(amount = amount, opponent = ally)
        opponent.heal_self(amount)


class burn(after_round_statuses):
    name = '烧伤'

    def effect(self, ally, opponent):
        '''烧伤状态触发时的效果，直接改变目标的hp'''
        amount = 10
        print(f'{ally.name}因{self.name}的影响')
        opponent.execute_damage(amount, ally)


class before_skill_statuses(Statuses):
    # 指技能释放前触发的状态
    def __init__(self, is_after_round = False,
                 is_before_skill = True,
                 is_during_skill = False,
                 is_after_skill = False):
        super().__init__(is_after_round,
                         is_before_skill,
                         is_during_skill,
                         is_after_skill)


class paralysis(before_skill_statuses):
    name = '麻痹'

    def effect(self, ally, opponent):
        '''麻痹状态触发时的效果，可能使目标 cannot_move'''
        if random.random() <= 0.25:
            cannot_move().apply(target = ally, duration = 1)


class during_skill_statuses(Statuses):
    # 指技能释放时触发的状态
    def __init__(self, is_after_round = False,
                 is_before_skill = False,
                 is_during_skill = True,
                 is_after_skill = False):
        super().__init__(is_after_round,
                         is_before_skill,
                         is_during_skill,
                         is_after_skill)

    def apply(self, target, duration):
        '''赋予状态'''
        has_this = False
        for i in target.statuses.keys():
               if i.name == self.name:
                   has_this = True
                   break

        if not has_this:
            target.statuses[self] = duration


class cannot_move(during_skill_statuses):
    name = 'cannot move'
    # 如果存在该状态，use_skill 直接被 pass
    # 详细见 use_skill


class after_charge(during_skill_statuses):
    name = 'after charge'

    def effect(self, ally: Pokemon, opponent: Pokemon, skill: Skills):
        print(f'{ally}对{opponent}使用了{skill}')
        skill.effect2(user = ally, opponent = opponent)


class after_skill_statuses(Statuses):
    # 指技能释放后触发的状态
    def __init__(self, is_after_round= False,
                 is_before_skill = False,
                 is_during_skill = False,
                 is_after_skill = True):
        super().__init__(is_after_round,
                         is_before_skill,
                         is_during_skill,
                         is_after_skill)


class damage_reduction(after_skill_statuses):
    name = '伤害减免'
    # 如果存在该状态，execute_damage 中的 amount 减免

    def __init__(self, is_after_round=False,
                 is_before_skill=False,
                 is_during_skill=False,
                 is_after_skill=True,
                 amount: float = None):
        super().__init__(is_after_round,
                         is_before_skill,
                         is_during_skill,
                         is_after_skill)
        self.amount = amount

    def apply(self, target, duration):
        '''赋予伤害减免状态'''
        reduction = 1 - self.amount
        for i in target.statuses.keys():
               if i.name == self.name:
                   reduction *= (1 - i.amount)
        self.amount = 1 - reduction
        target.statuses[self] = duration
