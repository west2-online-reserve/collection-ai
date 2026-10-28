from .base import FirePokemon
import skills

class Charmander(FirePokemon):

    def __init__(self) -> None:
        super().__init__(name = "小火龙", hp=110, attack=56, defense=21, sp_attack=56, sp_defense=25, dodge=10, speed=65)
        # 小火龙拥有改变天气的招式：大晴天
        self.skills = [skills.Ember(self.attack),
                       skills.FlameCharge(self.sp_attack),
                       skills.Flamethrower(self.sp_attack)
                       ]


class Vulpix(FirePokemon):

    def __init__(self) -> None:
        super().__init__(name = "六尾", hp=105, attack=41, defense=20, sp_attack=50, sp_defense=30, dodge=10, speed=66)
        self.skills = [skills.Ember(self.attack),
                       skills.WillOWisp(),
                       skills.FireBlast(self.sp_attack)
                       ]