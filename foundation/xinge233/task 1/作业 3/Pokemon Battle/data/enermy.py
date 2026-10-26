class Enermy_action:
    def __init__(self, enermy_pokemon, player_pokemon):
        self.enermy_pokemon = enermy_pokemon
        self.player_pokemon = player_pokemon
        self.show_list = []#听 ai 这样会比较安全，防止之后直接调用类导致出错

    def enermy_action_start(self):
        self.show_list = []
        from .action import Act
        import random

        self.enermy_random = random.randint(1, 100)
        self.enermy_action = Act(
            pokemon=self.enermy_pokemon, enermy_pokemon=self.player_pokemon
        )
        if (
            self.enermy_pokemon.__class__.__name__ == "Pikachu"
        ):  # 这里接着会写如果选择了各个精灵会调用不同函数
            return self.pikachu_action()
        elif self.enermy_pokemon.__class__.__name__ == "Charmander":
            return self.charmander_action()
        elif self.enermy_pokemon.__class__.__name__ == "Bulbasaur":
            return self.bulbasaur_action()
        elif self.enermy_pokemon.__class__.__name__ == "Squirtle":
            return self.squirtle_action()
        elif self.enermy_pokemon.__class__.__name__ == "MilkDragon":
            return self.milkdragon_action()

    def common_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )
        return self.show_list

    def pikachu_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )

        elif self.player_pokemon.element == "Water":
            self.show_list.extend(
                self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.player_pokemon.element == "Grass":
            self.show_list.extend(
                self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.enermy_random <= 50:
            self.show_list.extend(
                self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
            )
        else:
            self.show_list.extend(
                self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
            )
        return self.show_list

    def charmander_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )
        elif (
            self.player_pokemon.element == "Water"
            and self.player_pokemon.health
            <= 2 * (self.enermy_pokemon.attack - self.player_pokemon.defense)
        ):  # 敌人血量低于两倍攻击伤害时使用技能1
            self.show_list.extend(
                self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.player_pokemon.element == "Grass":
            if (
                self.enermy_random <= 10
                and self.enermy_pokemon.effect.get("burn", 0) == 0
            ):  # 在敌人没有烧伤的时候小概率使用技能2烧伤敌人
                self.show_list.extend(
                    self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
                )
            elif self.enermy_random <= 20:
                self.show_list.extend(
                    self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
                )
            else:
                self.show_list.extend(
                    self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
                )
        elif (
            self.player_pokemon.effect.get("burn", 0) == 1 and self.enermy_random <= 80
        ):  # 敌人烧伤即将结束大概率使用技能2烧伤敌人
            self.show_list.extend(
                self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
            )
        else:
            self.show_list.extend(
                self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
            )
        return self.show_list

    def bulbasaur_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.player_pokemon.element == "Fire":
            if self.enermy_pokemon.effect.get("poison", 0) == 0:  # 敌人没有中毒时使用技能1中毒敌人
                self.show_list.extend(
                    self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
                )
            elif self.enermy_random <= 50 and self.enermy_pokemon.effect.get("absorb", 0) == 0:  # 敌人没有吸收时大概率使用技能2吸收敌人血量
                self.show_list.extend(
                    self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
                )
            else:
                self.show_list.extend(
                    self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
                )
   

        elif self.player_pokemon.element == "Water":
            if self.player_pokemon.health < 2*(self.enermy_pokemon.attack - self.player_pokemon.defense):  # 敌人血量低于两倍攻击伤害时使用技能1
                self.show_list.extend(
                    self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
                )
            elif self.enermy_random<= 50 and self.enermy_pokemon.effect.get("absorb", 0) == 0:  # 敌人没有吸收时大概率使用技能1吸收敌人血量
                self.show_list.extend(
                    self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
                )
            elif self.enermy_random <= 80 and self.enermy_pokemon.effect.get("poison", 0) == 0:  # 敌人没有中毒时小概率使用技能1中毒敌人
                self.show_list.extend(
                    self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon)
                )
            else:
                self.show_list.extend(self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon))
        else:
            if self.enermy_random <= 60 and self.enermy_pokemon.effect.get("absorb", 0) == 0:  # 敌人没有吸收时大概率使用技能2吸收敌人血量
                self.show_list.extend(
                    self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon)
                )
            else:
                self.show_list.extend(
                    self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
                )
        return self.show_list
    def squirtle_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.player_pokemon.effect.get("Flame_Charge",0)== 1:
            self.show_list.extend(self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon))
        elif self.player_pokemon.element == "Electric":
            if self.enermy_random <= 30 :
                self.show_list.extend(self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon))
            else:
                self.show_list.extend(self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon))
        else:
            if self.enermy_random <= 30:
                self.show_list.extend(self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon))
            else:
                self.show_list.extend(self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon))
        return self.show_list
    def milkdragon_action(self):
        if (
            self.enermy_pokemon.health
            < self.player_pokemon.attack - self.enermy_pokemon.defense
        ):  # 普通攻击能删直接杀
            self.show_list.extend(
                self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon)
            )
        elif self.player_pokemon.element == "Grass":
            if self.enermy_random <= 30:
                self.show_list.extend(self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon))
            else:
                self.show_list.extend(self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon))
        else:
            if self.enermy_random <= 40 and self.enermy_pokemon.effect.get("MilkDragon_attack_up", [0, 0])[1] ==3 or self.enermy_pokemon.effect.get("MilkDragon_attack_up", [0, 0])[1] ==0:
                self.show_list.extend(self.enermy_pokemon.skill1(self.enermy_pokemon, self.player_pokemon))
            elif self.enermy_pokemon.effect.get("MilkDragon_attack_up", [0, 0])[1] <=1 and self.enermy_random <= 50:
                self.show_list.extend(self.enermy_pokemon.skill2(self.enermy_pokemon, self.player_pokemon))
            else:
                self.show_list.extend(self.enermy_action.fight(self.enermy_pokemon, self.player_pokemon))
        return self.show_list
            
            