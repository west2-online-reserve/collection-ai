from base.pokemon import Pokemon
from skills import DittoSkills


class NormalPokemon(Pokemon):
    type = "普通"

    def __init__(self, hp, attack, defense, dodge_chance):
        super().__init__(hp, attack, defense, dodge_chance)

    def type_effectiveness(self, opponent):
        return 1

    def begin(self):
        self.apply_status_effect()


class Ditto(NormalPokemon):
    name = "百变怪"

    def __init__(self, hp=70, attack=25, defense=15, dodge_chance=10):
        super().__init__(hp, attack, defense, dodge_chance)

    def initialize_skills(self):
        return [DittoSkills.Crash(), DittoSkills.Imitate()]
