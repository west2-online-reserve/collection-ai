from .base import WaterPokemon
import skills

class Squirtle(WaterPokemon):

    def __init__(self) -> None:
        super().__init__(name = "杰尼龟", hp=130, attack=45, defense=25, sp_attack=50, sp_defense=28, dodge=15, speed=43)
        self.skills = [skills.AquaJet(damage=self.attack),
                       skills.Shield(),
                       skills.Surf(self.sp_attack)
                       ]


class Psyduck(WaterPokemon):

    def __init__(self) -> None:
        super().__init__(name = "可达鸭", hp=135, attack=52, defense=28, sp_attack=50, sp_defense=20, dodge=15, speed=55)
        self.skills = [skills.Amnesia(),
                       skills.AquaJet(damage=self.attack),
                       skills.RainDance(),
                       skills.HydroPump(self.sp_attack)
                       ]