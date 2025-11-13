from base.skills import Skill
import random
from misc.tools import printWithDelay

SLEEP_TIME = 1


class Crash(Skill):
    name = "撞击"

    def __init__(self, amount: int = 1):
        self.amount = amount

    def execute(self, user, opponent):
        damage = user.attack * self.amount
        opponent.receive_damage(damage, self.name)


class Imitate(Skill):
    name = "模仿"

    def __init__(self):
        pass

    def execute(self, user, opponent):
        skills = opponent.skills
        skill_to_use = random.choice(skills)
        printWithDelay(f"{user.name} 模仿了 {opponent.name} 的 {skill_to_use.name}")
        user.use_skill(skill_to_use, opponent)
