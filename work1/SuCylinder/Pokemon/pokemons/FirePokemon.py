from base.pokemon import Pokemon
from skills import CharmanderSkills
from misc.tools import printWithDelay


class FirePokemon(Pokemon):
    type = "火"

    def __init__(self, hp, attack, defense, dodge_chance):
        super().__init__(hp, attack, defense, dodge_chance)
        self.base_attack = self.attack
        self.attack_increase = 1

    def begin(self):
        self.apply_status_effect()

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "草":
            printWithDelay("效果拔群!")
            effectiveness = 2.0
        elif opponent_type == "水":
            printWithDelay("收效甚微")
            effectiveness = 0.5
        return effectiveness

    def use_skill(self, skill, opponent):
        printWithDelay(f"{self.name} 使用了 {skill.name}!")
        if skill.execute(self, opponent):
            if self.attack_increase < 1.4:
                self.attack_increase += 0.1
                self.attack = self.base_attack * self.attack_increase
                printWithDelay(f"{self.name}攻击力增加")


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self, hp=80, attack=35, defense=15, dodge_chance=10) -> None:
        super().__init__(hp, attack, defense, dodge_chance)

    def initialize_skills(self):
        return [CharmanderSkills.Ember(), CharmanderSkills.Flame_Charge()]
