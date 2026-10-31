from __future__ import annotations

import random

from effects import Effect
from skills import (
    AquaJet,
    Ember,
    FlameCharge,
    ParasiticSeeds,
    QuickAttack,
    SeedBomb,
    Shield,
    Skill,
    Thunderbolt,
)


class Pokemon:
    name: str
    type: str

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_rate: int = 0,
    ) -> None:
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        self.skills = self.initialize_skills()
        self.alive = True
        self.statuses: list[Effect] = []
        self.shield_active = False
        self.paralyzed = False
        self.dodge_rate = dodge_rate
        self.charging_skill: Skill | None = None
        self._is_countering = False

    def initialize_skills(self) -> list[Skill]:
        raise NotImplementedError("具体宝可梦必须实现 initialize_skills()")

    def use_skill(
        self,
        skill: Skill,
        opponent: Pokemon,
        *,
        is_counter: bool = False,
    ) -> None:
        if not self.alive or not opponent.alive:
            return

        print(f"{self.name} 使用了 {skill.name}")
        hp_before = opponent.hp
        previous_counter_state = self._is_countering
        self._is_countering = previous_counter_state or is_counter

        try:
            skill.execute(self, opponent)
        finally:
            self._is_countering = previous_counter_state

        if opponent.hp < hp_before:
            self.increase_fire_attack()

    def heal_self(self, amount: int) -> int:
        if not self.alive:
            return 0

        amount = int(amount)
        hp_before = self.hp
        self.hp = min(self.max_hp, self.hp + max(0, amount))
        return self.hp - hp_before

    def try_dodge(
        self,
        attacker: Pokemon | None = None,
        bonus: int = 0,
    ) -> bool:
        if not self.alive:
            return False

        dodge_chance = min(100, max(0, self.dodge_rate + bonus))
        if random.random() >= dodge_chance / 100:
            return False

        print(f"{self.name}成功闪避了攻击！")
        if (
            attacker is not None
            and attacker.alive
            and not attacker._is_countering
        ):
            self.on_successful_dodge(attacker)
        return True

    def receive_damage(self, damage: int) -> int:
        if not self.alive:
            return 0

        actual_damage = max(0, int(damage) - self.defense)
        if actual_damage <= 0:
            print("宝可梦的防御抵挡了攻击！")
            return 0

        if self.shield_active:
            actual_damage = int(actual_damage * 0.5)
            self.shield_active = False
            print(f"{self.name}的护盾使本次伤害减少了50%！")

        return self._lose_hp(actual_damage)

    def receive_status_damage(self, damage: int) -> int:
        """Apply fixed status damage without defense or elemental passives."""
        if not self.alive:
            return 0
        return self._lose_hp(max(0, int(damage)), show_damage=False)

    def _lose_hp(self, damage: int, *, show_damage: bool = True) -> int:
        actual_damage = min(self.hp, damage)
        self.hp = max(0, self.hp - damage)

        if show_damage:
            print(f"{self.name}受到了{actual_damage}点伤害")
            print(f"剩余HP：{self.hp} / {self.max_hp}")

        if self.hp <= 0:
            self.alive = False
            print(f"{self.name}倒下了")
        return actual_damage

    def add_status_effect(self, effect: Effect) -> None:
        if effect.duration <= 0:
            return
        for index, status in enumerate(self.statuses):
            if type(status) is type(effect):
                self.statuses[index] = effect
                return
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        if not self.alive:
            return

        for status in self.statuses[:]:
            if status.duration <= 0:
                self.statuses.remove(status)
                print(f"{status.name}状态结束了！")
                continue
            if not self.alive:
                break
            status.apply(self)
            status.decrease_duration()

            if status.duration <= 0 and status in self.statuses:
                self.statuses.remove(status)
                print(f"{status.name}状态结束了！")

    def type_effectiveness(self, opponent: Pokemon) -> float:
        raise NotImplementedError("属性类必须实现 type_effectiveness()")

    def increase_fire_attack(self) -> None:
        return None

    def on_successful_dodge(self, attacker: Pokemon) -> None:
        return None

    def begin(self) -> None:
        """Run this Pokemon's end-of-turn elemental passive."""
        return None


class GrassPokemon(Pokemon):
    type = "Grass"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Water":
            return 2.0
        if opponent.type == "Fire":
            return 0.5
        return 1.0

    def begin(self) -> None:
        self.grass_attribute()

    def grass_attribute(self) -> None:
        amount = int(self.max_hp * 0.1)
        healed = self.heal_self(amount)
        if healed > 0:
            print(f"{self.name}触发草属性被动，恢复了{healed}点HP！")


class Bulbasaur(GrassPokemon):
    name = "Bulbasaur"

    def __init__(
        self,
        hp: int = 100,
        attack: int = 35,
        defense: int = 10,
        dodge_rate: int = 10,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self) -> list[Skill]:
        return [SeedBomb(), ParasiticSeeds()]


class Chikorita(GrassPokemon):
    name = "Chikorita"

    def __init__(
        self,
        hp: int = 90,
        attack: int = 30,
        defense: int = 15,
        dodge_rate: int = 15,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self) -> list[Skill]:
        return [SeedBomb(damage=35), ParasiticSeeds(amount=8)]


class WaterPokemon(Pokemon):
    type = "Water"

    def receive_damage(self, damage: int) -> int:
        if self.alive and random.random() < 0.5:
            damage = int(damage * 0.7)
            print(f"{self.name}触发水属性被动，减免了30%伤害！")
        return super().receive_damage(damage)

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Fire":
            return 2.0
        if opponent.type == "Electric":
            return 0.5
        return 1.0


class Squirtle(WaterPokemon):
    name = "Squirtle"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 25,
        defense: int = 20,
        dodge_rate: int = 20,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self) -> list[Skill]:
        return [AquaJet(), Shield()]


class FirePokemon(Pokemon):
    type = "Fire"

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_rate: int = 0,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)
        self.base_attack = attack
        self.attack_boost_stacks = 0

    def increase_fire_attack(self) -> None:
        if self.attack_boost_stacks >= 4:
            return
        self.attack_boost_stacks += 1
        self.attack = int(
            self.base_attack * (1 + 0.1 * self.attack_boost_stacks)
        )
        print(f"{self.name}的攻击力提升了！当前攻击力：{self.attack}")

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Grass":
            return 2.0
        if opponent.type == "Water":
            return 0.5
        return 1.0


class Charmander(FirePokemon):
    name = "Charmander"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 15,
        dodge_rate: int = 10,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self) -> list[Skill]:
        return [Ember(), FlameCharge()]


class ElectricPokemon(Pokemon):
    type = "Electric"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "Water":
            return 2.0
        if opponent.type == "Grass":
            return 0.5
        return 1.0

    def on_successful_dodge(self, attacker: Pokemon) -> None:
        if not self.alive or not attacker.alive or not self.skills:
            return

        skill = random.choice(self.skills)
        print(f"{self.name}触发电属性被动，立即反击！")
        self.use_skill(skill, attacker, is_counter=True)


class PikaChu(ElectricPokemon):
    name = "PikaChu"

    def __init__(
        self,
        hp: int = 80,
        attack: int = 35,
        defense: int = 5,
        dodge_rate: int = 30,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)

    def initialize_skills(self) -> list[Skill]:
        return [Thunderbolt(), QuickAttack()]
