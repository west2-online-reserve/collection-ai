import random
import skills
import effects


class Pokemon:
    name: str
    max_hp: int
    hp: int
    type: str
    attack: int
    defense: int

    def __init__(
        self,
        name: str,
        max_hp: int,
        type: str,
        attack: int,
        defense: int,
        dodge_rate: float,
    ) -> None:
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.type = type
        self.attack = attack
        self.defense = defense
        self.alive = True
        self.dodge_rate = max(0.0, min(1.0, float(dodge_rate)))
        self.status_effects = []  # 存储状态效果
        self.paralyzed = False  # 麻痹状态标志
        self.reduce = 0.0  # 减伤率
        self.skills = self.intialize_skills()
        self.nerve_damage = False
        self.nerve_count = 0

    def __str__(self) -> str:
        return f"{self.name} ({self.type}) HP:{self.hp}/{self.max_hp}"

    def intialize_skills(self):
        # 这里可以根据 Pokemon 的类型或其他属性来初始化技能
        pass

    def use_skill(self, skill, opponent):
        # 使用技能攻击对手
        if self.paralyzed:
            print(f"{self.name} 的麻痹效果正在生效，无法使用技能！")
            self.paralyzed = False  # 麻痹效果持续一回合，下一回合恢复正常
            return
        print(f"{self.name} uses {skill.name} on {opponent.name}!")
        skill.execute(self, opponent)

    def dodge_is_successful(self) -> bool:
        if random.random() < self.dodge_rate:
            print(f"{self.name} 闪避了这次伤害")
            return True
        return False

    def nerve_is_successful(self):
        if self.nerve_damage:
            return
        if self.nerve_count >= 3:
            print(f"{self.name}进入神经损伤状态！")
            self.receive_damage(55, source="nerve")
            self.status_effects.append(effects.NerveDamage())
            self.nerve_damage = True
        else:
            return

    def receive_damage(self, damage, opponent, source="normal"):
        # 神经损伤来源的伤害无法被闪避，并且无视防御和减伤
        if source == "nerve":
            print(f"{self.name} 受到了神经损伤的伤害，无法闪避！")
            self.hp -= damage*self.type_effectiveness(opponent)
            self.hp = max(0, self.hp)
            print(f"{self.name} 受到了 {damage} 点神经损伤真伤！剩余 HP: {self.hp}")
            if self.hp <= 0:
                self.alive = False
                print(f"{self.name} 晕倒了！")
            return
        # 躲闪判定：若触发则本次伤害完全无效
        if self.dodge_is_successful():
            return

        # 计算实际伤害并更新 HP
        actual_damage = max(0, damage - self.defense)
        actual_damage = max(0, actual_damage - int(actual_damage * self.reduce))
        self.hp -= actual_damage*self.type_effectiveness(opponent)
        self.hp = max(0, self.hp)
        print(f"{self.name} 受到了 {actual_damage} 点伤害！剩余 HP: {self.hp}")
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} 晕倒了！")

    def heal_self(self, amount):
        # 恢复 HP
        self.hp += amount
        print(f"{self.name} 恢复了 {amount} HP! 现在的 HP: {self.hp}")

    def set_dodge_rate(self, rate):
        # 统一限制躲闪率范围，防止配置错误
        self.dodge_rate = max(0.0, min(1.0, float(rate)))

    def add_status_effect(self, effect):
        # 添加状态效果
        self.status_effects.append(effect)
        print(f"{self.name} 受到了 {effect.name} 的影响！")

    def apply_status_effects(self):
        # 应用所有状态效果
        for effect in self.status_effects:
            effect.apply(self)
            effect.decrease_duration()
        # 移除持续时间为 0 的效果
        self.status_effects = [
            effect for effect in self.status_effects if effect.duration > 0
        ]

        # 神经损伤效果结束后，重置对应状态与层数
        has_nerve_damage = any(
            isinstance(effect, effects.NerveDamage) for effect in self.status_effects
        )
        #判断神经损伤是否在这一次的效果结算中结束，如果结束则把状态重置
        if not has_nerve_damage and self.nerve_damage:
            self.nerve_damage = False
            self.nerve_count = 0
            print(f"{self.name} 的神经损伤状态解除了。")

    def is_alive(self):
        return self.alive

    def type_effectiveness(self, opponent):
        # 计算类型相克关系，返回伤害倍数
        pass

    def begin(self):
        # 每回合开始时应用状态效果
        pass


class GrassPokemon(Pokemon):
    type = "Grass"

    def type_effectiveness(self, opponent) -> float:
        if opponent.type == "Water":
            return 2.0  # 草系对水系伤害加倍
        elif opponent.type == "Fire":
            return 0.5  # 草系对火系伤害减半
        else:
            return 1.0  # 其他情况正常伤害

    def begin(self):
        # 每回合开始时应用状态效果
        self.apply_status_effects()
        self.grass_ability()

    def grass_ability(self):
        amount = self.max_hp * 0.1  # 恢复 10% 的最大 HP
        if self.hp < self.max_hp:
            if self.hp + amount > self.max_hp:
                self.hp = self.max_hp
            else:
                self.hp += amount
            print(f"{self.name} 恢复了 {amount} HP! 现在的 HP: {self.hp}/{self.max_hp}")


class WaterPokemon(Pokemon):
    type = "Water"

    def type_effectiveness(self, opponent) -> float:
        if opponent.type == "Fire":
            return 2.0  # 水系对火系伤害加倍
        elif opponent.type == "Grass":
            return 0.5  # 水系对草系伤害减半
        else:
            return 1.0  # 其他情况正常伤害

    def begin(self):
        # 每回合开始时应用状态效果
        self.apply_status_effects()

    def receive_damage(self, damage, source="normal"):
        # 水系被动：受到伤害时，有 50% 的几率减免 30% 的伤害
        if source != "nerve" and random.random() < 0.5:
            print(f"{self.name} 的 Water Passive 激活了！伤害减免 30%")
            damage = damage * 0.7  # 减免 30%，也就是只承受 70% 的原始伤害

        super().receive_damage(damage, source=source)


class FirePokemon(Pokemon):
    type = "Fire"

    def __init__(
        self, name: str, max_hp: int, attack: int, defense: int, dodge_rate: float
    ) -> None:
        super().__init__(name, max_hp, "Fire", attack, defense, dodge_rate)
        self.fire_ability_active = 0
        self.base_attack = attack

    def type_effectiveness(self, opponent) -> float:
        if opponent.type == "Grass":
            return 2.0  # 火系对草系伤害加倍
        elif opponent.type == "Water":
            return 0.5  # 火系对水系伤害减半
        else:
            return 1.0  # 其他情况正常伤害

    def begin(self):
        # 每回合开始时应用状态效果
        self.apply_status_effects()

    def use_skill(self, skill, opponent):
        super().use_skill(skill, opponent)
        if self.fire_ability_active > 0:
            amount = self.base_attack * 0.1 * self.fire_ability_active
            self.attack += amount
            print(f"{self.name} 的 Fire Passive 激活了！攻击力增加了 {amount}")
        if self.fire_ability_active < 4:
            self.fire_ability_active += 1


class ElectricPokemon(Pokemon):
    type = "Electric"

    def type_effectiveness(self, opponent) -> float:
        if opponent.type == "Water":
            return 2.0  # 电系对水系伤害加倍
        elif opponent.type == "Grass":
            return 0.5  # 电系对草系伤害减半
        else:
            return 1.0  # 其他情况正常伤害

    def begin(self):
        # 每回合开始时应用状态效果
        self.apply_status_effects()

    def electric_ability(self, skill, opponent):
        if self.dodge_is_successful():
            print(
                f"{self.name} 的 Electric Passive 激活了！反击对手"
            )
            super().use_skill(skill, opponent)


class Bulbasaur(GrassPokemon):
    def __init__(self, dodge_rate: float = 0.1) -> None:
        super().__init__(
            name="Bulbasaur",
            max_hp=100,
            type="Grass",
            attack=35,
            defense=10,
            dodge_rate=dodge_rate,
        )

    def intialize_skills(self):
        # 初始化 Bulbasaur 的技能
        return [skills.SeedBomb(damage=self.attack), skills.ParasiticSeeds(amount=10)]


class Squirtle(WaterPokemon):
    def __init__(self, dodge_rate: float = 0.2) -> None:
        super().__init__(
            name="Squirtle",
            max_hp=80,
            type="Water",
            attack=25,
            defense=20,
            dodge_rate=dodge_rate,
        )

    def intialize_skills(self):
        # 初始化 Squirtle 的技能
        return [skills.AquaJet(damage=self.attack * 1.4), skills.Shield(reduce=0.5)]


class Charmander(FirePokemon):
    def __init__(self, dodge_rate: float = 0.1) -> None:
        super().__init__(
            name="Charmander", max_hp=80, attack=35, defense=15, dodge_rate=dodge_rate
        )
        self.count = 0

    def intialize_skills(self):
        # 初始化 Charmander 的技能
        return [
            skills.Ember(damage=self.attack, chance=0.1),
            skills.FlameCharge(damage=self.attack * 3, reduce=0.8),
        ]


class Pikachu(ElectricPokemon):
    def __init__(self, dodge_rate: float = 0.3) -> None:
        super().__init__(
            name="Pikachu",
            max_hp=80,
            type="Electric",
            attack=35,
            defense=5,
            dodge_rate=dodge_rate,
        )

    def intialize_skills(self):
        # 初始化 Pikachu 的技能
        return [
            skills.Thunderbolt(damage=self.attack * 1.4, chance=0.1),
            skills.Quickattack(damage=self.attack, chance=0.1),
        ]


class PhonoR(ElectricPokemon):
    # 初始数值很低，靠打爆条来造成高伤害
    name = "PhonoR"

    def __init__(self, dodge_rate: float = 0.3) -> None:
        super().__init__(
            name=self.name,
            max_hp=70,
            type="Electric",
            attack=20,
            defense=10,
            dodge_rate=dodge_rate,
        )

    def intialize_skills(self):
        # 初始化 PhonoR 的技能
        return [
            skills.Echoism(amount=1),
            skills.Mass(damage=self.attack, extra_damage=20),
        ]
