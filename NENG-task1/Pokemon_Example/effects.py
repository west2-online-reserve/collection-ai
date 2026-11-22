import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(self, duration: int) -> None:
        # 初始化效果持续时间
        self.duration = duration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        self.duration -= 1
        time.sleep(0.4)
        print(f"{self.name} 效果剩余持续回合：{self.duration}")


class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, duration: int = 3) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        damage = int(pokemon.max_hp * 0.1)
        time.sleep(0.5)
        print(f"{pokemon.name} 受到了 {damage} 点中毒伤害！")
        pokemon.receive_damage(damage, is_status=True)


class HealEffect(Effect):
    name = "治疗"

    def __init__(self, amount: int, duration: int = 3) -> None:
        super().__init__(duration)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)


class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int = 10, duration: int = 2, attacker: "Pokemon | None" = None) -> None:
        super().__init__(duration)
        self.damage = damage
        self.attacker = attacker

    def apply(self, pokemon: "Pokemon") -> None:
        hp_before = pokemon.hp
        time.sleep(0.5)
        print(f"{pokemon.name} 受到了 {self.damage} 点烧伤伤害！")
        pokemon.receive_damage(self.damage, is_status=True)

        if (
            pokemon.hp < hp_before
            and self.attacker is not None
            and self.attacker.alive
            and hasattr(self.attacker, "fire_stacks")
            and self.attacker.fire_stacks < 4
        ):
            self.attacker.fire_stacks += 1
            self.attacker.attack = int(self.attacker.base_attack * (1 + 0.1 * self.attacker.fire_stacks))
            time.sleep(0.5)
            print(
                f"{self.attacker.name} 的烧伤触发了火系被动！"
                f"攻击力提升至 {self.attacker.attack}（{self.attacker.fire_stacks}/4 层）"
            )


class ParalysisEffect(Effect):
    name = "麻痹"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.is_paralyzed = True
        time.sleep(0.5)
        print(f"{pokemon.name} 处于麻痹状态，本回合无法行动！")


class ShieldEffect(Effect):
    name = "护盾"

    def __init__(self, duration: int = 1) -> None:
        super().__init__(duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.has_shield = True
