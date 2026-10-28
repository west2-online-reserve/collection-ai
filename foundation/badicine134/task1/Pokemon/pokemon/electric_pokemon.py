from .base import ElectricPokemon
import skills

class Pikachu(ElectricPokemon):

    def __init__(self) -> None:
        super().__init__(name = "皮卡丘", hp=100, attack=55, defense=18, sp_attack=50, sp_defense=20, dodge=20, speed=80)
        self.skills = [skills.Thunderbolt(self.sp_attack),
                       skills.QuickAttack(self.attack),
                       skills.ThunderPunch(self.attack)
                       ]


class Magnemite(ElectricPokemon):

    def __init__(self) -> None:
        super().__init__(name = "小磁怪", hp=90, attack=35, defense=20, sp_attack=95, sp_defense=25, dodge=20, speed=45)
        self.skills = [skills.IronDefense(),
                       skills.QuickAttack(self.attack),
                       skills.ThunderstormCall(),
                       skills.Thunder(self.sp_attack)
                       ]