import random
from typing import Optional

from enums import Type
from . import Skill, Buff


class Pokemon:
    def __init__(
        self,
        name: str = "",
        health_point: int = 0,
        type_: Type = Type.DARK,
        damage: int = 0,
        defense: int = 0,
        evasion_rate: int = 0,
        skills: Optional[list[Skill]] = None,
        bot: bool = False,
    ) -> None:
        self.name: str = name
        self.max_health_point: int = health_point
        self._health_point: int = health_point
        self.type: Type = type_
        self.damage: int = damage
        self.defense: int = defense
        self.evasion_rate: int = evasion_rate
        self.skills: list[Skill] = skills
        self.buffs: list[Buff] = []
        self.skip_turn: bool = False
        self.damage_reduction: int = 0
        self.miss_bonus: int = 0
        self.bot: bool = bot
        self.enemy: Optional["Pokemon"] = None
        self.name = self.name + (" (电脑)" if self.bot else "")

    @property
    def health_point(self):
        return self._health_point

    @health_point.setter
    def health_point(self, value: int):
        self._health_point = max(value, 0)
        self._health_point = min(self._health_point, self.max_health_point)

    @property
    def dead(self) -> bool:
        return self.health_point <= 0

    def set_enemy(self, enemy: "Pokemon"):
        self.enemy: "Pokemon" = enemy

    def start_turn(self) -> None:
        self.skip_turn = False
        self.damage_reduction = False
        self.miss_bonus = 0
        self.on_turn_start()
        for buff in self.buffs:
            buff.apply(self)
            buff.decrease_duration()

        self.buffs = [buff for buff in self.buffs if buff.duration != 0]

        if self.skip_turn:
            return

        self.select_skill()

    def perform_skill(self, skill: Skill) -> None:
        if skill.current_turns == skill.turns_required:
            print(f"[{self.name}]正在施展「{skill.name}」...")
            skill.perform(self, self.enemy)
            skill.current_turns = 0
        else:
            skill.current_turns += 1
            print(
                f"[{self.name}]正在准备「{skill.name}」({skill.current_turns}/{skill.turns_required})..."
            )

    def attacked(self, pokemon: "Pokemon", damage: int) -> bool:
        pokemon.on_pre_attack()
        self.on_pre_attacked()
        if self.is_missed():
            print(f"[{self.name}]躲开了攻击!")
            self.on_mise_attack()
            return False

        super_effective = pokemon.is_super_effective()
        not_very_effective = self.is_super_effective()

        if super_effective:
            damage *= 2
            print("效果拔群，", end="")

        elif not_very_effective:
            damage /= 2
            print("效果不佳，", end="")

        damage = max(0, damage - self.defense)

        damage = int((100 - self.damage_reduction) / 100.0 * damage)

        self.health_point -= damage
        print(f"[{self.name}]受到了{damage}点伤害! (剩余HP: {self.health_point})")
        pokemon.on_post_attack()
        self.on_post_attacked()

        return True

    def check_dead(self) -> bool:
        if self.dead:
            print(f"[{self.name}]寄了, [{self.enemy.name}]获胜!")
            return True
        return False

    def is_missed(self) -> bool:
        return random.randint(1, 100) <= self.evasion_rate + self.miss_bonus

    def add_buff(self, buff: Buff):
        self.buffs = [b for b in self.buffs if b.name != buff.name]
        self.buffs.append(buff)

    def is_super_effective(self) -> bool:
        match self.type:
            case Type.FIRE:
                return self.enemy.type == Type.GRASS
            case Type.GRASS:
                return self.enemy.type == Type.WATER
            case Type.WATER:
                return self.enemy.type == Type.FIRE
            case Type.ELECTRIC:
                return self.enemy.type == Type.WATER
            case Type.DARK:
                return True

    def select_skill(self) -> None:

        if self.bot:
            skill = random.choice(self.skills)
        else:
            while True:
                try:
                    skill_options = [
                        f"{index + 1}. {skill.name}"
                        for index, skill in enumerate(self.skills)
                    ]
                    choice = input(
                        f"你的[{self.name}]的技能: \n"
                        + f"\n".join(skill_options)
                        + "\n"
                    )
                    index = int(choice) - 1
                    if index >= len(self.skills) or index < 0:
                        print("你选择的选项无效，请重新选择！")
                        continue
                    skill = self.skills[index]
                    break
                except (ValueError, TypeError):
                    print("你选择的选项无效，请重新选择！")

        # noinspection PyUnboundLocalVariable
        self.perform_skill(skill)

    def on_pre_attack(self) -> None: ...

    def on_post_attack(self) -> None: ...

    def on_mise_attack(self) -> None: ...

    def on_pre_attacked(self) -> None: ...

    def on_post_attacked(self) -> None: ...

    def on_turn_start(self) -> None: ...
