from __future__ import annotations
import skills
from skills import Skill
from effects import Effect
import random


class Pokemon:
    name: str
    type: str

    def __init__(
        self, hp: int, attack: int, defense: int, speed: int, evasion_chance: int
    ) -> None:
        self.hp: int = hp
        self.max_hp: int = hp
        self.attack: float = attack
        self.defense: int = defense
        self.skills: list[Skill] = self.initialize_skills()
        self.alive: bool = True
        self.statuses: list[Effect] = []
        self.speed: float = speed
        self.action: bool = True
        self.damage_free: float = 1
        self.charging: bool = False
        self.charge_turn: int = 0
        self.chargingskill: Skill | None = None
        self.evasion_chance: int = evasion_chance
        self.evasion: bool = False
        self.is_counterattack: bool = False  # 判断是否处于反击状态

    def initialize_skills(self) -> list[Skill]:
        raise NotImplementedError

    def use_skill(self, skill: Skill, opponent: Pokemon) -> int:
        if self.chargingskill is None:
            damage = skill.execute(self, opponent)
            return damage
        elif self.charge_turn > 0:
            print(f"{self.name}正在蓄力")
            self.charge_turn -= 1
            return 0
        else:
            assert self.chargingskill is not None
            damage = self.chargingskill.apply(self, opponent)
            self.chargingskill = None
            self.charging = False
            return damage

    def heal_self(self, amount: int) -> None:
        if not isinstance(amount, int):
            amount = int(amount)
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(f"{self.name}回复{amount}点血量! 血量: {self.hp}/{self.max_hp}")

    def receive_damage(
        self,
        attack: float = 0,
        power: float = 0,
        effective: float = 1,
        fixed_damage: float = 0,
        evation_effetive: float = 1,
    ) -> int:
        if fixed_damage != 0:
            final_damage: int = int(fixed_damage)
        else:
            if random.randint(1, 100) <= self.evasion_chance * evation_effetive:
                print(f"{self.name}闪避了攻击")
                if not self.is_counterattack:
                    self.evasion = True
                return 0

            final_damage = int(attack * power * effective * self.damage_free)
            final_damage -= self.defense
            self.damage_free = 1
            if final_damage <= 0:
                print(f"{self.name}的防御吸收了攻击")
                self.damage_free = 1
                return 0

        initial_hp = self.hp
        self.hp -= final_damage
        if self.hp > 0:
            print(
                f"{self.name}受到{final_damage}点伤害! 剩余血量: {self.hp}/{self.max_hp}"
            )
        else:
            self.alive = False
            print(f"{self.name}濒死了")
            return initial_hp

        return final_damage

    def add_status_effect(self, effect: Effect) -> None:
        if effect in self.statuses:
            print(f"{self.name}已处于{effect.name}效果")
            return
        if effect.type == self.type:
            print(f"{effect.name}对{self.type}无效")
        self.statuses.append(effect)

    def apply_status_effect(self, opponent: Pokemon) -> None:
        for status in self.statuses[:]:
            status.apply(self, opponent)
            status.decrease_duration()
            if status.duration <= 0:
                print(f"{self.name}的{status.name}效果结束了")
                self.statuses.remove(status)

    def on_evade(self, attacker: Pokemon, skill: Skill) -> None:
        pass

    def judge_evasion(self) -> bool | None:
        pass

    def begin(self) -> None:
        pass

    def __str__(self) -> str:
        return f"{self.name} 属性: {self.type}"


class GrassPokemon(Pokemon):
    type: str = "草"

    def begin(self) -> None:
        self.grass_attribute()

    def grass_attribute(self) -> None:
        amount = self.max_hp * 0.1
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
        print(
            f"{self.name}在回合开始回复{amount}点血量  当前血量: {self.hp}/{self.max_hp}"
        )


class FirePokemon(Pokemon):
    type: str = "火"

    def __init__(
        self, hp: int, attack: int, defense: int, speed: int, evasion_chance: int
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)
        self.stack: int = 0

    def use_skill(self, skill: Skill, opponent: Pokemon) -> int:
        damage = super().use_skill(skill, opponent)
        if damage > 0 and self.stack < 4:
            self.stack += 1
            self.attack *= 1.1
            print(f"火系特性发动，攻击力提升！当前层数: {self.stack}")
        return damage


class WaterPokemon(Pokemon):
    type: str = "水"

    def receive_damage(
        self,
        attack: float = 0,
        power: float = 0,
        effective: float = 1,
        fixed_damage: float = 0,
        evation_effetive: float = 1,
    ) -> int:
        if fixed_damage != 0:
            final_damage = int(fixed_damage)
        else:
            if random.randint(1, 100) <= self.evasion_chance * evation_effetive:
                print(f"{self.name}闪避了攻击")
                if not self.is_counterattack:
                    self.evasion = True
                return 0

            final_damage = int(attack * power * effective * self.damage_free)
            final_damage -= self.defense
            self.damage_free = 1
            if final_damage <= 0:
                print(f"{self.name}的防御吸收了攻击")
                self.damage_free = 1
                return 0

        if random.randint(1, 100) <= 50:
            final_damage = int(final_damage * 0.7)
            print(f"{self.name}特性发动，减免了伤害")

        initial_hp = self.hp
        self.hp -= final_damage
        if self.hp > 0:
            print(
                f"{self.name}受到{final_damage}点伤害! 剩余血量: {self.hp}/{self.max_hp}"
            )
        else:
            self.alive = False
            print(f"{self.name}濒死了")
            return initial_hp

        return final_damage


class RockPokemon(Pokemon):
    type: str = "岩石"

    def __init__(
        self, hp: int, attack: int, defense: int, speed: int, evasion_chance: int
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)
        self.been_death: bool = False

    def receive_damage(
        self,
        attack: float = 0,
        power: float = 0,
        effective: float = 1,
        fixed_damage: float = 0,
        evation_effetive: float = 1,
    ) -> int:
        if fixed_damage != 0:
            final_damage = fixed_damage
        else:
            if random.randint(1, 100) <= self.evasion_chance * evation_effetive:
                print(f"{self.name}闪避了攻击")
                if not self.is_counterattack:
                    self.evasion = True
                return 0
            final_damage = int(attack * power * effective * self.damage_free)
            final_damage -= self.defense
            self.damage_free = 1
            if final_damage <= 0:
                print(f"{self.name}的防御吸收了攻击")
                self.damage_free = 1
                return 0

        initial_hp = self.hp
        self.hp -= final_damage
        if self.hp > 0:
            print(
                f"{self.name}受到{final_damage}点伤害! 剩余血量: {self.hp}/{self.max_hp}"
            )
        else:
            if self.been_death == False:
                if random.randint(1, 100) <= 50:
                    self.alive = False
                    print(f"{self.name}濒死了")
                    return initial_hp
                else:
                    self.hp = 1
                    print(
                        f"{self.name}的坚韧特性发动！剩余血量：{self.hp}/{self.max_hp}"
                    )
                    self.been_death = True
                    return initial_hp - 1
            else:
                self.alive = False
                print(f"{self.name}濒死了")
                return initial_hp
        return final_damage


class ElectricPokemon(Pokemon):
    type: str = "电"

    def on_evade(self, attacker: Pokemon, skill: Skill) -> None:
        print(f"触发{self.name}的静电特性")
        if skill is not None:
            self.is_counterattack = True  # 增加标记，使反击不能被反击
            self.use_skill(skill=skill, opponent=attacker)
            self.is_counterattack = False  # 反击结束取消标记

    def judge_evasion(self) -> bool:
        if self.evasion == True:
            return True
        return False


class Bulbasaur(GrassPokemon):
    name: str = "妙蛙种子"

    def __init__(
        self,
        hp: int = 100,
        attack: int = 50,
        defense: int = 10,
        speed: int = 30,
        evasion_chance: int = 10,
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)

    def initialize_skills(self) -> list[Skill]:
        return [skills.SeedBomb(), skills.ParasiticSeeds()]


class Pikachu(ElectricPokemon):
    name: str = "皮卡丘"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 5,
        speed: int = 40,
        evasion_chance: int = 30,
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)

    def initialize_skills(self) -> list[Skill]:
        return [skills.Thunderbolt(), skills.QuickAttack()]


class Squirtle(WaterPokemon):
    name: str = "杰尼龟"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 25,
        defense: int = 20,
        speed: int = 15,
        evasion_chance: int = 20,
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)

    def initialize_skills(self) -> list[Skill]:
        return [skills.AquaJet(), skills.Shield()]


class Charmander(FirePokemon):
    name: str = "小火龙"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 15,
        speed: int = 20,
        evasion_chance: int = 10,
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)

    def initialize_skills(self) -> list[Skill]:
        return [skills.Ember(), skills.FlameCharge()]


class Onix(RockPokemon):
    name: str = "大岩蛇"

    def __init__(
        self,
        hp: int = 100,
        attack: int = 30,
        defense: int = 30,
        speed: int = 15,
        evasion_chance: int = 15,
    ) -> None:
        super().__init__(hp, attack, defense, speed, evasion_chance)

    def initialize_skills(self) -> list[Skill]:
        return [skills.HeadSmash(), skills.RockTomb()]


Pokemon_Pool: list[type[Pokemon]] = [Bulbasaur, Pikachu, Squirtle, Charmander, Onix]
