from base.pokemon import Pokemon
from skills import PikaChuSkills
from misc.tools import printWithDelay


class ElectricPokemon(Pokemon):
    type = "电"

    def __init__(self, hp, attack, defense, dodge_chance):
        self.is_dodged = False
        super().__init__(hp, attack, defense, dodge_chance)

    def type_effectiveness(self, opponent: Pokemon):
        effectiveness = 1.0
        opponent_type = opponent.type

        if opponent_type == "水":
            printWithDelay("效果拔群!")
            effectiveness = 2.0
        elif opponent_type == "草":
            printWithDelay("收效甚微")
            effectiveness = 0.5
        return effectiveness

    def dodged(self):
        self.is_dodged = super().dodged()
        return self.is_dodged

    def begin(self):
        self.apply_status_effect()


class PikaChu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self, hp=80, attack=35, defense=5, dodge_chance=30) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(hp, attack, defense, dodge_chance)

    def initialize_skills(self):
        return [PikaChuSkills.Thunderbolt(), PikaChuSkills.Quick_Attack()]
