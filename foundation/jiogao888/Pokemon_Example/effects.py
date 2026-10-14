from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon


class Effect:
    name: str

    def __init__(
        self,
        duration: int,
        number: int = 1,
        noduration: bool = False,
        max_number: int = 1,
    ) -> None:
        # 初始化效果持续时间
        self.base_duration = duration
        self.duration = duration
        self.number = number
        self.max_number = max_number
        self.noduration = noduration

    def apply(self, pokemon: "Pokemon") -> None:
        # 应用效果的抽象方法，子类需要实现
        raise NotImplementedError

    def decrease_duration(self) -> None:
        # 减少效果持续时间
        if not (self.noduration):
            self.duration -= 1
            print(f"{self.name} effect duration decreased. Remaining: {self.duration}")

    def reload(self) -> None:
        self.duration = self.base_duration
        print(f"{self.name}'duration is reload")


class BuffEffect(Effect):
    def __init__(
        self,
        duration,
        amount: int,
        number=1,
        noduration=False,
        max_number=1,
    ):
        super().__init__(duration, number, noduration, max_number)
        self.amount = amount


class HitEffect(Effect):
    pass


class AttbuffEffect(BuffEffect):
    pass


class DRbuffEffect(BuffEffect):
    pass


class DefbuffEffect(BuffEffect):
    pass


class PoisonEffect(HitEffect):
    name = "Poison"

    def __init__(self, damage: int = 10, duration: int = 3, max_number=1) -> None:
        super().__init__(duration=duration, max_number=max_number)
        self.damage = damage

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.hp_lost(self.damage)
        print(f"{pokemon.name} takes {self.damage} poison damage!")


class HealEffect(HitEffect):
    name = "Heal"

    def __init__(self, amount: int = 10, duration: int = 3, max_number=1) -> None:
        super().__init__(duration=duration, max_number=max_number)
        self.amount = amount

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.heal_self(self.amount)
        print(f"{pokemon.name} heals {self.amount} HP!")


class attraise(AttbuffEffect):
    name = "attack raise"

    def __init__(
        self,
        amount: int = 10,
        number: int = 1,
        duration: int = 100,
        noduration: bool = True,
        max_number=4,
    ):
        super().__init__(
            duration=duration,
            amount=amount,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )


class fire(HitEffect):
    name = "fire"

    def __init__(self, duration: int = 2):
        super().__init__(duration=duration)

    def apply(self, pokemon: "Pokemon") -> None:
        pokemon.hp_lost(10)
        print(f"{pokemon.name} takes 10 fire damage!")


class Shield(DRbuffEffect):
    name = "Shield"

    def __init__(self, duration=1, number=1, noduration=False, amount=50, max_number=1):
        super().__init__(
            duration=duration,
            amount=amount,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )


class paralysis(HitEffect):
    name = "paralysis"

    def __init__(self, duration=1, number=1, noduration=False, max_number=1):
        super().__init__(
            duration=duration,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )

    def apply(self, pokemon: "Pokemon"):
        pass


class RageDRup(DRbuffEffect):
    name = "Rage Damade reduction up"

    def __init__(
        self, duration=100, amount=15, number=1, noduration=True, max_number=3
    ):
        super().__init__(
            duration=duration,
            amount=amount,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )

    def apply(self, pokemon: "Pokemon"):
        pass


class Rageattup(AttbuffEffect):
    name = "Rage attack up"

    def __init__(
        self, duration=100, amount=15, number=1, noduration=True, max_number=3
    ):
        super().__init__(
            duration=duration,
            amount=amount,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )

    def apply(self, pokemon: "Pokemon"):
        pass


class Ragedefup(DefbuffEffect):
    name = "Rage defense up"

    def __init__(
        self, duration=100, amount=15, number=1, noduration=True, max_number=3
    ):
        super().__init__(
            duration=duration,
            amount=amount,
            number=number,
            noduration=noduration,
            max_number=max_number,
        )

    def apply(self, pokemon: "Pokemon"):
        pass
