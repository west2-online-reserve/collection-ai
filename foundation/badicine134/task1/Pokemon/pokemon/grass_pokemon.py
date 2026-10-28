from .base import GrassPokemon
import skills

class Bulbasaur(GrassPokemon):

    def __init__(self) -> None:
        # 初始化 Bulbasaur 的属性
        super().__init__(name = "妙蛙种子", hp=120, attack=54, defense=26, sp_attack=65, sp_defense=30, dodge=10, speed=45)
        self.skills = [skills.SeedBomb(self.attack),
                       skills.ParasiticSeeds(amount=10),
                       skills.MegaDrain(self.sp_attack)
                       ]


class Bellsprout(GrassPokemon):

    def __init__(self) -> None:
        super().__init__(name = "喇叭芽", hp=120, attack=55, defense=18, sp_attack=75, sp_defense=20, dodge=10, speed=75)
        self.skills = [skills.VineWhip(self.sp_attack),
                       skills.PoisonPowder(),
                       skills.SunnyDay(),
                       skills.RazorLeaf(self.attack)
                       ]