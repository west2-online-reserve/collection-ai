import random
import time
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon"):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name}"


class SeedBomb(Skill):
    name = "种子炸弹"

    def __init__(self, activation_chance: int = 15) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        damage = user.attack * 1.0 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False

        if hit and opponent.alive and random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect(duration=3))
            time.sleep(0.5)
            print(f"{opponent.name} 因 {self.name} 进入了中毒状态！")


class ParasiticSeeds(Skill):
    name = "寄生种子"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        heal_amount = int(opponent.max_hp * 0.1)
        user.add_status_effect(effects.HealEffect(heal_amount, duration=3))
        opponent.add_status_effect(effects.PoisonEffect(duration=3))
        time.sleep(0.5)
        print(f"{user.name} 对 {opponent.name} 种下了 {self.name}！")


class Thunderbolt(Skill):
    name = "十万伏特"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        damage = user.attack * 1.4 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False

        if hit and opponent.alive and random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect(duration=1))
            time.sleep(0.5)
            print(f"{opponent.name} 因 {self.name} 进入了麻痹状态！")


class QuickAttack(Skill):
    name = "电光一闪"

    def __init__(self, double_hit_chance: int = 10) -> None:
        super().__init__()
        self.double_hit_chance = double_hit_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        damage = user.attack * 1.0 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False

        if opponent.alive and random.randint(1, 100) <= self.double_hit_chance:
            time.sleep(0.5)
            print(f"{user.name} 连续发动了第二次 {self.name}！")
            opponent.receive_damage(damage)


class AquaJet(Skill):
    name = "水枪"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        damage = user.attack * 1.4 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False


class Shield(Skill):
    name = "护盾"

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        user.has_shield = True
        user.add_status_effect(effects.ShieldEffect(duration=1))
        time.sleep(0.5)
        print(f"{user.name} 使用了 {self.name}，下回合受到的伤害减少 50%！")


class Ember(Skill):
    name = "火花"

    def __init__(self, activation_chance: int = 10) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        damage = user.attack * 1.0 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False

        if hit and opponent.alive and random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect(damage=10, duration=2, attacker=user))
            time.sleep(0.5)
            print(f"{opponent.name} 因 {self.name} 进入了烧伤状态！")


class FlameCharge(Skill):
    name = "蓄能爆炎"

    def __init__(self, activation_chance: int = 80) -> None:
        super().__init__()
        self.activation_chance = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> None:
        if getattr(user, "charging_skill", None) != self:
            user.charging_skill = self
            time.sleep(0.5)
            print(f"{user.name} 正在为 {self.name} 蓄力！")
            return

        user.charging_skill = None
        damage = user.attack * 3.0 * user.type_effectiveness(opponent)
        hit = opponent.receive_damage(damage, extra_shanbi=20)

        if not hit and opponent.type == "Electric" and opponent.alive and user.alive and not getattr(user, "_is_countering", False):
            time.sleep(0.5)
            print(f"{opponent.name} 触发了电系被动，立即进行反击！")
            opponent._is_countering = True
            opponent.use_skill(random.choice(opponent.skills), user)
            opponent._is_countering = False

        if hit and opponent.alive and random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect(damage=10, duration=2, attacker=user))
            time.sleep(0.5)
            print(f"{opponent.name} 因 {self.name} 进入了烧伤状态！")
class pengfu(Skill):

    name = "捧腹大笑"

    def execute(self,user:"Pokemon",opponent:"Pokemon") -> None:
        print(f"{user.name}发出了{self.name}！")
        print('''
        ⠀⠄⠀⠀⢠⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⡟⠀⠀⠀⠀⠀⢀⡤⠄⣀⡀⠀⠀⠀⠀⠀⠉⠓⠢⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣼⠥⠀⠀⠀⠀⠀⣽⣾⠿⠿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠒⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠻⣿⠷⡄⠀⠀⠀⠉⠁⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠙⣦⣇⣀⠤⠤⣶⣺⡽⠋⢻⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢨⠿⣯⡛⠛⣉⠁⠀⠀⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢠⠋⠀⠈⠳⣄⣿⣀⣀⣼⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢠⠏⠀⠀⠀⠀⠈⠳⣭⣭⢾⡡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⡀⠀⠀⠀⠀⠀
⠀⠀⢀⡟⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠀
⠀⠀⣸⠃⠀⠀⠀⠘⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠚⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀
⠀⠀⡟⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡄⠀⠀
⠀⢸⡇⠶⠷⠯⠅⡀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢄⣤⠿⠿⣤⡀⠀⠀⠀⠀⠀⠀⠀⢹⡄⠀
⠀⠈⣇⠀⠀⠀⠀⢠⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠉⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⢷⠀
⠀⠀⢻⡄⠀⠀⠀⠀⢿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡤⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠁⠀⠀⠀⠀⢸⠀
⠀⠀⠀⠹⣄⠀⠀⠀⢸⠽⢒⡚⢻⡆⠀⢰⡟⠿⠻⠂⣐⣬⠎⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⢨⠀
⠀⠀⠀⠀⠈⠳⣅⠀⡏⠀⠲⢤⣻⣤⣀⡤⠉⠡⠂⠀⠀⠀⠀⠀⠀⠀⣀⡠⠴⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀
⠀⠀⠀⠀⠀⠀⠈⠳⣦⠀⠀⠰⣿⣏⣴⣰⡦⠀⠀⠀⠀⠀⠠⣤⠶⠟⠁⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠈⢷⢦⣰⣿⣿⡿⣟⣟⡥⡄⣀⡤⠴⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠃⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣝⠿⠿⠛⠋⠉⠛⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⣀⠀⠀⡼⣃⣤⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⢱⣾⣶⢱⡞⣝⢰⣶⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠁⠀⠀⠈⠓⠦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠫⠭⠽⢺⠁⠘⠪⠭⠀
'''
)
        time.sleep(0.5)
        user.hp = user.max_hp
        user.has_shield = True
        user.add_status_effect(effects.ShieldEffect(duration=1))
        print("奶娃大人已恢复到满状态！")
        time.sleep(0.5)
        print('''⠀
⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠚⠛⠋⠛⠛⠶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⣿⠀⠀⢠⣾⢿⣷⡄⠨⢳⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣃⣂⣀⣈⠻⠾⠿⠃⠀⠀⢻⡄⠀⠀⠀⠀⠀⠀⠀
⠀⠆⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⠛⠛⠛⠀⠀⠀⠀⠀⠀⠃⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣾⣹⣵⡽⣦⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⢷⡄⠀⠀⠀⠀⠀⠀⠀
⠀⠻⣮⡫⢀⢿⠸⢧⣤⣄⣀⡀⠀⠀⢀⣀⣰⣤⡴⠾⠉⠐⠀⠀⠀⠀⠠⠀⠀⠀⠀⠂⠀⠐⠻⣄⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠻⢦⡶⠬⠁⠀⠂⠭⠙⠛⠻⠩⠍⠐⠂⠀⠀⠀⠀⠀⠄⡀⡐⠂⠀⠀⠀⠀⠀⠀⡄⠀⠙⣦⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠷⣅⠀⡀⠐⠀⢈⠆⠀⢄⡴⣣⠤⠚⠛⠛⠒⠲⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠈⢷⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠛⠶⣥⣀⡀⠤⢄⡿⠙⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⢄⠀⠀⣦⡔⡁⣬⠃⠀⠈⢷⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⣻⠟⠀⠀⠀⠀⠀⠀⠀⡀⢀⣀⣀⡀⠀⠀⠀⣠⣽⠧⠶⠦⠀⠀⠈⢸⡇⢠⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡏⠀⠀⠀⠄⠀⠀⠀⢀⢰⣯⡛⠋⠙⣛⡃⠈⠠⡄⠀⠀⠄⢄⠀⢐⢸⡇⠈⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠈⠀⠀⠀⠀⠀⠀⠀⠐⢘⣿⣦⣤⡀⠆⡚⠀⠀⠀⠀⠘⠄⠘⣋⣫⠟⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠐⠀⠀⠀⠈⠀⠙⠐⠨⣤⣍⡛⠛⠓⠒⠚⠛⠙⠫⢹⡆⠀⠐⠀
⠀⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⠠⠃⠠⠀⠁⠀⠀⠀⣼⠃⠀⠐⠀
⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢘⣇⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⢀⡿⠀⠀⠀⠀
⠀⡆⠀⣀⠀⠀⠀⠀⠀⢀⣠⠶⢫⠏⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢄⡾⠁⠀⠀⠐⠀
⠀⡇⠀⠀⠀⠀⠀⠀⣴⠫⠓⠉⠀⡀⡐⠙⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡈⠀⢸⠀
⠀⡇⠀⠀⠀⠀⠀⡰⣧⠀⠄⠐⢀⠀⢀⠀⠀⠉⠓⠦⢄⡀⠀⠀⠁⠀⢀⣀⣀⠀⠀⠀⠀⠀⠀⠐⠀⡑⢝⣮⢸⠀
⠀⡇⠠⠀⠀⠄⠀⠀⠹⣮⢄⠀⠀⢰⢶⡶⠶⠶⢛⠛⠛⠛⠛⠛⢛⠛⠶⠶⠶⣤⣄⣀⠀⠄⠀⢀⡰⠐⠸⣸⣇⠀
⠀⡇⠀⠀⠀⠀⠀⠀⠀⠈⢷⡕⣀⠸⢸⡇⠁⠀⠀⡤⠤⠤⠤⠌⠁⠀⠀⢀⣀⡐⠊⡩⢛⢳⡐⣋⢡⡁⣠⡟⢸⠀
⠀⠓⠒⠐⠒⠀⠀⠀⠒⠀⠐⠹⠎⠂⠿⠐⠀⠒⠒⠀⠂⠀⠀⠂⠓⠊⠈⠓⠒⠐⠂⠐⠒⠸⠇⠃⠢⠾⠋⠒⠘⠀
''')
        return

class yipie(Skill):
    name = "惊鸿一瞥"

    def execute(self,user:"Pokemon",opponent:"Pokemon" ) -> None:
        print(f"{user.name}{self.name}")
        time.sleep(0.5)
        if user.numpie < 4:
            user.numpie += 1
        if user.numpie == 4:
            opponent.hp = -10
            print(f"臣服在奶娃大人的淫威之下吧!")
            time.sleep(0.5)
            print('''⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⠤⠤⠤⠤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⢀⡴⠒⠉⠒⠂⠀⠀⠀⠀⠀⠀⠀⠀⠙⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣼⡅⠀⠀⠀⢸⣴⢠⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⠀⠀⠈⠻⢿⠿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣃⣀⣀⣀⣒⣒⠾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢻⠛⠛⠛⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⡄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⠀
⠀⠀⠀⠀⠀⢀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄
⠀⠀⠀⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱
⠀⠀⠀⡴⢃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈
                                                    ''')
        return