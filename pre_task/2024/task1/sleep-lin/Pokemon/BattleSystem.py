'''这个文档是战斗系统'''
import Pokemon,copy,random

d = {'属性': '被克制', '草': '火', '火': '水', '水': '电', '电': '草', '毒': '水','无':'无'}
d1 = {'属性': '克制', '草': '电', '火': '草', '水': '火', '电': '水', '毒': '草','无':'无'}

class Battle:
    def __init__(self, player_team, computer_team) -> None:
        self.player_team = copy.deepcopy(player_team)  # 深拷贝玩家队伍
        self.computer_team = copy.deepcopy(computer_team)  # 深拷贝电脑队伍
        self.player_pokemon = None  # 玩家上场的宝可梦
        self.computer_pokemon = None  # 电脑上场的宝可梦
        self.player_win = False
        self.computer_win = False
        self.skip = False
        self.team_battle()  # 开始队伍战斗

    def team_battle(self):
        while len(self.player_team) > 0 and len(self.computer_team) > 0:
            # 选择玩家和电脑的宝可梦
            if not self.player_pokemon or self.player_pokemon.is_fainted():
                self.player_pokemon = self.player_team.choose_pokemon()
            if not self.computer_pokemon or self.computer_pokemon.is_fainted():
                self.computer_pokemon = self.computer_team.cup_choose_pokemon()
            # 进行队员之间的战斗
            self.player_battle()

            # 检查队员比赛结束后的状态，移除已晕倒的宝可梦
            if self.player_pokemon.is_fainted():
                print(f"{self.player_pokemon.name} 昏厥了！")
                self.player_team.remove_pokemon(self.player_pokemon)
            if self.computer_pokemon.is_fainted():
                print(f"{self.computer_pokemon.name} 昏厥了！")
                self.computer_team.remove_pokemon(self.computer_pokemon)

        # 判断最终胜负
        if len(self.player_team) == 0:
            print("电脑赢得了队伍战斗！")
            self.computer_win = True
        elif len(self.computer_team) == 0:
            print("玩家赢得了队伍战斗！")
            self.player_win = True

    def player_battle(self):
        round_count = 0
        while not self.player_pokemon.is_fainted() and not self.computer_pokemon.is_fainted():
            round_count += 1
            print(f"\n=== 回合 {round_count} ===")

            # 玩家经历三个流程
            self.start_of_round(self.player_pokemon)
            if self.player_pokemon.is_fainted():
                break
            if self.skip == False:
                self.player_round()
                #特判电属性的额外技能
                if self.computer_pokemon.extra_skill == True:
                    self.computer_round()
                    self.computer_pokemon.extra_skll = False
                    if self.player_pokemon.is_fainted():
                        break
                self.end_of_round(self.player_pokemon)
            self.skip = False

            # 判断电脑是否昏厥
            if self.computer_pokemon.is_fainted():
                break
            print(f"\n=== 电脑回合 ===")
            # 电脑经历三个流程
            self.start_of_round(self.computer_pokemon)
            if self.computer_pokemon.is_fainted():
                break
            if self.skip == False:
                self.computer_round()
                #特判电属性的额外技能
                if self.player_pokemon.extra_skill == True:
                    self.player_round()
                    self.player_pokemon.extra_skll = False
                    if self.computer_pokemon.is_fainted():
                        break
                self.end_of_round(self.computer_pokemon)
            self.skip = False
        
    def start_of_round(self, pokemon):
        # 第一个检查状态：是否有特殊状态
        opponent_pokemon = self.computer_pokemon if pokemon == self.player_pokemon else self.player_pokemon
        if '蓄能炎爆'in opponent_pokemon.status_effects.keys():
            print('因为在蓄能，所以跳过出招阶段')
            self.skip = True
        if pokemon.status_effects:
            for effect, duration in list(pokemon.status_effects.items()):
                print(f"{pokemon.name} 受到了 {effect} 状态的影响，持续时间剩余 {duration} 回合")
                self.suffer_effect(effect,pokemon)
                if duration > 1:
                    pokemon.status_effects[effect] -= 1
                else:
                    if effect == '蓄能炎爆':
                        print('蓄能炎爆爆发了！生命值减少，并可能被烧伤')
                        pokemon.take_damage(opponent_pokemon.attack*3,'火')
                        if random.random() < 0.8:
                            print(f"{pokemon.name} 被烧伤了，每回合会受到额外伤害！")
                            pokemon.apply_status_effect('烧伤',100)
                    del pokemon.status_effects[effect]

    def suffer_effect(self, effect, pokemon):
    # 根据当前宝可梦找到对手的宝可梦
        opponent_pokemon = self.computer_pokemon if pokemon == self.player_pokemon else self.player_pokemon
        if effect == '麻痹':
            print(f'{pokemon.name} 因为麻痹，跳过出招阶段')
            self.skip = True
        elif effect == '中毒':
            print(f'{pokemon.name} 因为中毒，生命值减少')
            damage = pokemon.max_hp // 10
            pokemon.apply_damage(damage)
        elif effect == '烧伤':
            print(f'{pokemon.name} 因为烧伤，生命值减少')
            pokemon.apply_damage(10)
        elif effect == '寄生':
            print(f'{pokemon.name} 因为寄生，生命值减少，{opponent_pokemon.name} 生命值恢复')
            damage = pokemon.max_hp // 10
            pokemon.apply_damage(damage)
            opponent_pokemon.hp+=damage  # 为对手回血
        elif effect == '双重打击':
            print('因为上回合对方触发了双重打击，生命值减少')
            pokemon.take_damage(opponent_pokemon.attack,'电')
        

    def player_round(self):
        chosen_skill = self.player_pokemon.choose_skill()
        chosen_skill.deal_damage(self.player_pokemon, self.computer_pokemon)
        if self.computer_pokemon.dodge_attack == False:
            chosen_skill.apply_effect(self.player_pokemon, self.computer_pokemon)
        else :
            self.computer_pokemon.dodge_attack = False
        chosen_skill.change(self.player_pokemon,d[self.computer_pokemon.attribute])

    def computer_round(self):
        chosen_skill = self.computer_pokemon.cpu_choose_skill()
        chosen_skill.deal_damage(self.computer_pokemon, self.player_pokemon)
        if self.player_pokemon.dodge_attack == False:
            chosen_skill.apply_effect(self.computer_pokemon, self.player_pokemon)
        else:
            self.player_pokemon.dodge_attack = False
        chosen_skill.change(self.computer_pokemon,d[self.player_pokemon.attribute])

    def end_of_round(self, pokemon):
        # 第三个流程：是否触发被动技能
        pokemon.passive_skills()