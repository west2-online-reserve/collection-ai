from __future__ import annotations

import random

from effects import Effect
from skills import (
    AquaJet,
    Bubble,
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
    name: str = "宝可梦"
    type: str = "一般"

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_rate: int,
    ) -> None:
        if hp <= 0 or attack <= 0 or defense < 0:
            raise ValueError("宝可梦的 HP、攻击力和防御力不合法")
        if not 0 <= dodge_rate <= 100:
            raise ValueError("闪避率必须在 0 到 100 之间")

        self.max_hp = hp
        self.hp = hp
        self.base_attack = attack
        self.attack = attack
        self.defense = defense
        self.dodge_rate = dodge_rate
        self.alive = True
        self.statuses: list[Effect] = []
        self.skills: list[Skill] = self.initialize_skills()
        self.shield_active = False
        self.paralyzed = False
        self.flinched = False
        self.charging_skill: Skill | None = None
        self.in_counter = False

    def initialize_skills(self) -> list[Skill]:
        raise NotImplementedError("子类必须实现 initialize_skills()")

    def type_effectiveness(self, opponent: Pokemon) -> float:
        raise NotImplementedError("属性类必须实现 type_effectiveness()")

    def use_skill(
        self,
        skill: Skill,
        opponent: Pokemon,
        *,
        is_counter: bool = False,
    ) -> None:
        if not self.alive or not opponent.alive:
            return

        print(f"{self.name} 使用了 {skill.name}！")
        hp_before = opponent.hp
        self.in_counter = is_counter
        try:
            skill.execute(self, opponent)
        finally:
            self.in_counter = False

        if opponent.hp < hp_before:
            self.on_deal_damage()

    def try_dodge(self, attacker: Pokemon | None = None, extra_dodge: int = 0) -> bool:
        if not self.alive:
            return False
        chance = max(0, min(100, self.dodge_rate + extra_dodge))
        if random.randint(1, 100) > chance:
            return False

        print(f"{self.name} 成功闪避了攻击！")
        if attacker is not None and not attacker.in_counter:
            self.on_dodge(attacker)
        return True

    def receive_damage(self, damage: int) -> int:
        if not self.alive or damage <= 0:
            return 0

        damage = self.modify_incoming_damage(damage)
        actual = damage - self.defense
        if actual <= 0:
            print(f"{self.name} 的防御抵挡了这次攻击！")
            return 0

        if self.shield_active:
            actual = max(1, int(actual * 0.5))
            self.shield_active = False
            print(f"{self.name} 的护盾使伤害减半！")

        return self._lose_hp(actual)

    def receive_status_damage(self, damage: int) -> int:
        if not self.alive or damage <= 0:
            return 0
        return self._lose_hp(damage)

    def _lose_hp(self, damage: int) -> int:
        actual = min(self.hp, damage)
        self.hp -= actual
        print(f"{self.name} 受到了 {actual} 点伤害！剩余 HP：{self.hp}/{self.max_hp}")
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
            print(f"{self.name} 昏厥了！")
        return actual

    def heal_self(self, amount: int) -> int:
        if not self.alive or amount <= 0:
            return 0
        healed = min(amount, self.max_hp - self.hp)
        if healed <= 0:
            return 0
        self.hp += healed
        print(f"{self.name} 恢复了 {healed} 点 HP！剩余 HP：{self.hp}/{self.max_hp}")
        return healed

    def add_status_effect(self, effect: Effect) -> None:
        if not self.alive:
            return
        for index, existing in enumerate(self.statuses):
            if type(existing) is type(effect):
                self.statuses[index] = effect
                print(f"{self.name} 的{effect.name}状态被刷新了。")
                return
        self.statuses.append(effect)

    def apply_status_effect(self) -> None:
        if not self.alive:
            return
        for effect in self.statuses[:]:
            if not self.alive:
                break
            effect.apply(self)
            effect.decrease_duration()
            if effect.duration <= 0 and effect in self.statuses:
                self.statuses.remove(effect)
                print(f"{self.name} 的{effect.name}状态解除了。")

    def modify_incoming_damage(self, damage: int) -> int:
        return damage

    def on_deal_damage(self) -> None:
        pass

    def on_dodge(self, attacker: Pokemon) -> None:
        pass

    def begin(self) -> None:
        pass

    def __str__(self) -> str:
        return f"{self.name}（{self.type}属性）"


class GrassPokemon(Pokemon):
    type = "草"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "水":
            return 2.0
        if opponent.type == "火":
            return 0.5
        return 1.0

    def begin(self) -> None:
        if self.hp >= self.max_hp:
            return
        amount = max(1, int(self.max_hp * 0.1))
        print(f"{self.name} 触发了草属性被动！")
        self.heal_self(amount)


class FirePokemon(Pokemon):
    type = "火"

    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_rate: int,
    ) -> None:
        super().__init__(hp, attack, defense, dodge_rate)
        self.attack_stacks = 0

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "草":
            return 2.0
        if opponent.type == "水":
            return 0.5
        return 1.0

    def on_deal_damage(self) -> None:
        if self.attack_stacks >= 4:
            return
        self.attack_stacks += 1
        self.attack = int(self.base_attack * (1 + 0.1 * self.attack_stacks))
        print(
            f"{self.name} 触发了火属性被动，攻击力提升到 {self.attack}！"
            f"（{self.attack_stacks}/4 层）"
        )


class WaterPokemon(Pokemon):
    type = "水"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "火":
            return 2.0
        if opponent.type == "电":
            return 0.5
        return 1.0

    def modify_incoming_damage(self, damage: int) -> int:
        if random.random() < 0.5:
            print(f"{self.name} 触发了水属性被动，减免 30% 伤害！")
            return max(1, int(damage * 0.7))
        return damage


class ElectricPokemon(Pokemon):
    type = "电"

    def type_effectiveness(self, opponent: Pokemon) -> float:
        if opponent.type == "水":
            return 2.0
        if opponent.type == "草":
            return 0.5
        return 1.0

    def on_dodge(self, attacker: Pokemon) -> None:
        if not self.alive or not attacker.alive or not self.skills:
            return
        skill = random.choice(self.skills)
        print(f"{self.name} 触发了电属性被动，立刻反击！")
        self.use_skill(skill, attacker, is_counter=True)


class PikaChu(ElectricPokemon):
    name = "皮卡丘"

    def __init__(self) -> None:
        super().__init__(hp=80, attack=35, defense=5, dodge_rate=30)

    def initialize_skills(self) -> list[Skill]:
        return [Thunderbolt(), QuickAttack()]


class Bulbasaur(GrassPokemon):
    name = "妙蛙种子"

    def __init__(self) -> None:
        super().__init__(hp=100, attack=35, defense=10, dodge_rate=10)

    def initialize_skills(self) -> list[Skill]:
        return [SeedBomb(), ParasiticSeeds()]


class Squirtle(WaterPokemon):
    name = "杰尼龟"

    def __init__(self) -> None:
        super().__init__(hp=80, attack=25, defense=20, dodge_rate=20)

    def initialize_skills(self) -> list[Skill]:
        return [AquaJet(), Shield()]


class Charmander(FirePokemon):
    name = "小火龙"

    def __init__(self) -> None:
        super().__init__(hp=80, attack=35, defense=15, dodge_rate=10)

    def initialize_skills(self) -> list[Skill]:
        return [Ember(), FlameCharge()]


class Totodile(WaterPokemon):
    name = "小锯鳄"

    def __init__(self) -> None:
        super().__init__(hp=90, attack=30, defense=12, dodge_rate=15)

    def initialize_skills(self) -> list[Skill]:
        return [Bubble(), AquaJet()]
