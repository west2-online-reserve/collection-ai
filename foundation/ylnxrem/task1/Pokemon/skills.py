import random
from typing import TYPE_CHECKING
import effects

if TYPE_CHECKING:
    from pokemon import Pokemon


class Skill:
    name: str
    type: str
    priority: int

    def __init__(self) -> None:
        pass

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        print(f"{user.name}使用了{self.name}")
        return 0  # 在子类中会返回造成的实际伤害

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        return 1.0

    def __str__(self) -> str:
        return f"{self.name}"


class GrassSkill(Skill):
    type: str = "草"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective = 1.5
        if opponent.type in ["电", "草", "火"]:
            effective *= 0.5
        if opponent.type in ["水", "岩石"]:
            effective *= 2
        return effective


class ElectricSkill(Skill):
    type: str = "电"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective *= 1.5
        if opponent.type in ["电", "草"]:
            effective *= 0.5
        if opponent.type in ["水"]:
            effective *= 2
        return effective


class NormalSkill(Skill):
    type: str = "一般"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective *= 1.5
        return effective


class WaterSkill(Skill):
    type: str = "水"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective *= 1.5
        if opponent.type in ["电", "草", "水"]:
            effective *= 0.5
        if opponent.type in ["火", "岩石"]:
            effective *= 2
        return effective


class FireSkill(Skill):
    type: str = "火"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective *= 1.5
        if opponent.type in ["火", "水", "岩石"]:
            effective *= 0.5
        if opponent.type in ["草"]:
            effective *= 2
        return effective


class Rockskill(Skill):
    type: str = "岩石"

    def __init__(self) -> None:
        super().__init__()

    def cal_skill_effective(self, user: "Pokemon", opponent: "Pokemon") -> float:
        effective: float = 1.0
        if self.type == user.type:
            effective *= 1.5
        if opponent.type in ["岩石", "草", "水"]:
            effective *= 0.5
        if opponent.type in ["火"]:
            effective *= 2
        return effective


class SeedBomb(GrassSkill):
    name: str = "种子炸弹"
    priority: int = 0

    def __init__(self, power: float = 1.05, activation_chance: int = 15) -> None:
        super().__init__()
        self.power: float = power
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.PoisonEffect())
            print(f"{self.name}使{opponent.name}中毒!")
        return damage


class ParasiticSeeds(GrassSkill):
    name: str = "寄生种子"
    priority: int = 0

    def __init__(self, activation_chance: int = 60) -> None:
        super().__init__()
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParasiticEffect())
            print(f"{opponent.name}被寄生")
        else:
            print(f"{self.name}没有命中{opponent.name}")
        return 0


class Thunderbolt(ElectricSkill):
    name: str = "十万伏特"
    priority: int = 0

    def __init__(self, power: float = 1.4, activation_chance: int = 10) -> None:
        super().__init__()
        self.power: float = power
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.ParalysisEffect())
            print(f"{opponent.name}被麻痹")
        return damage


class QuickAttack(NormalSkill):
    name: str = "电光一闪"
    priority: int = 1

    def __init__(self, power: float = 1.0, activation_chance: int = 10) -> None:
        super().__init__()
        self.power: float = power
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        if random.randint(1, 100) <= self.activation_chance:
            print("触发二次攻击")
            damage += opponent.receive_damage(
                attack=user.attack, power=self.power, effective=effective
            )
        return damage


class AquaJet(WaterSkill):
    name: str = "水枪"
    priority: int = 0

    def __init__(self, power: float = 1.2) -> None:
        super().__init__()
        self.power: float = power

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        return damage


class Shield(WaterSkill):
    name: str = "护盾"
    priority: int = 0

    def __init__(self) -> None:
        super().__init__()

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        user.damage_free *= 0.5
        return 0


class Ember(FireSkill):
    name: str = "火花"
    priority: int = 0

    def __init__(self, power: float = 1.2, activation_chance: int = 10) -> None:
        super().__init__()
        self.power: float = power
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}被灼伤")
        return damage


class FlameCharge(FireSkill):
    name: str = "蓄能焰袭"
    priority: int = 0

    def __init__(self, power: float = 3, activation_chance: int = 10) -> None:
        super().__init__()
        self.power: float = power
        self.activation_chance: int = activation_chance

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        user.charging = True
        user.charge_turn = 0
        user.chargingskill = self
        return 0

    def apply(self, user: "Pokemon", opponent: "Pokemon") -> int:
        print(f"蓄力结束，{user.name}使用{self.name}攻击{opponent.name}")
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack,
            power=self.power,
            effective=effective,
            evation_effetive=1.2,
        )
        if random.randint(1, 100) <= self.activation_chance:
            opponent.add_status_effect(effects.BurnEffect())
            print(f"{opponent.name}被灼伤")
        return damage


class RockTomb(Rockskill):
    name: str = "岩石封锁"
    priority: int = 0

    def __init__(self, power: float = 1.0) -> None:
        super().__init__()
        self.power: float = power

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        opponent.speed *= 0.8
        return damage


class HeadSmash(Rockskill):
    name: str = "双刃头槌"
    priority: int = 0

    def __init__(self, power: float = 3) -> None:
        super().__init__()
        self.power: float = power

    def execute(self, user: "Pokemon", opponent: "Pokemon") -> int:
        super().execute(user, opponent)
        effective = self.cal_skill_effective(user, opponent)
        damage = opponent.receive_damage(
            attack=user.attack, power=self.power, effective=effective
        )
        if damage != 0:
            print(f"{user.name}受到反伤")
            user.receive_damage(fixed_damage=damage * 0.25)
        return damage
