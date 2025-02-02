from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon

# 定义效果类
class Effect:
    name: str

# 初始化效果持续时间
    def __init__(self, duration: int):
        self.duration = duration

# 抽象方法,子类应具体实现效果
    def apply(self, pokemon: "Pokemon"):
        raise NotImplementedError
# 减少效果持续时间
    def decrease_duration(self):
        self.duration -= 1
        print(f"{self.name}还剩下{self.duration}轮")

# 麻痹
class ParalysisEffect(Effect):
    name = "麻痹"

    def __init__(self, duration: int):
        super().__init__(duration)

    def apply(self, pokemon:"Pokemon"):
       pass

# 中毒
class PoisonEffect(Effect):
    name = "中毒"

    def __init__(self, damage: int, duration: int = 3):
        super().__init__(duration)
        self.damage = damage

    def apply(self, pokemon: "Pokemon"):
        pokemon.hp -= (pokemon.max_hp * self.damage / 100)
        pokemon.hp = round(pokemon.hp, 2)
        print(f"{pokemon.name}受到了{round(pokemon.max_hp * self.damage / 100, 2)}点中毒伤害!现在的血量为{pokemon.hp}/{pokemon.max_hp}HP.")
        if pokemon.hp <= 0:
            pokemon.alive = False
            print(f"{pokemon.name}因为中毒效果而昏厥了!")

# 寄生
class ParasitismEffect(Effect):
    name = "寄生"

    def __init__(self, numeric_value: int, duration: int = 3):
        super().__init__(duration)
        self.numeric_value = numeric_value

    def apply(self, pokemon):
        pokemon.hp -= (pokemon.max_hp * self.numeric_value / 100)
        pokemon.hp = round(pokemon.hp, 2)
        print(f"{pokemon.name}因被寄生损失了{round(pokemon.max_hp * self.numeric_value / 100, 2)}点HP!现在的血量为{pokemon.hp}/{pokemon.max_hp}HP.")
        if pokemon.hp <= 0:
            pokemon.alive = False
            print(f"{pokemon.name}因为寄生效果而昏厥了!")
        
# 治疗
class HealEffect(Effect):
    name = "治疗"

    def __init__(self, numeric_value: int, duration: int = 3):
        super().__init__(duration)
        self.numeric_value = numeric_value

    def apply(self, pokemon: "Pokemon"):
        pokemon.heal_self(self.numeric_value)

# 护盾
class ShieldEffect(Effect):
    name = "护盾"

    def __init__(self, numeric_value: int, duration: int):
        super().__init__(duration)
        self.numeric_value = numeric_value

    def apply(self, pokemon: "Pokemon"):
        pass

# 烧伤
class BurnEffect(Effect):
    name = "烧伤"

    def __init__(self, damage: int, duration: int = 2):
        super().__init__(duration)
        self.damage = damage
    
    def apply(self, pokemon: "Pokemon"):
        pokemon.hp -= self.damage
        pokemon.hp = round(pokemon.hp, 2)
        print(f"{pokemon.name}受到了{self.damage}点烧伤伤害!现在的血量为{pokemon.hp}/{pokemon.max_hp}HP.")
        if pokemon.hp <= 0:
            pokemon.alive = False
            print(f"{pokemon.name}因为烧伤效果而昏厥了!")


