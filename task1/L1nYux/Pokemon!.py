import random
from time import sleep

# 宝可梦属性配置
POKEMON_CONFIG = {
    "皮卡丘": {"HP": 85, "ATK": 22, "DEF": 7, "AVD": 0.2, "TYP": 1, "LEFT": 1},
    "杰尼龟": {"HP": 90, "ATK": 20, "DEF": 8, "AVD": 0.1, "TYP": 2, "LEFT": 2},
    "小火龙": {"HP": 80, "ATK": 24, "DEF": 7, "AVD": 0.1, "TYP": 3, "LEFT": 1},
    "妙蛙种子": {"HP": 90, "ATK": 20, "DEF": 9, "AVD": 0.1, "TYP": 4, "LEFT": 1},
    "卡比兽": {"HP": 110, "ATK": 15, "DEF": 9, "AVD": 0.05, "TYP": 3, "LEFT": 2}
}

POKEMON_LIST = list(POKEMON_CONFIG.keys())
TYPE_NAMES = {1: "电", 2: "水", 3: "火", 4: "草"}


def printer(strings, delay=0.05):
    """逐字输出文本"""
    for char in strings:
        print(char, end='', flush=True)
        sleep(delay)
    print()


class Pokemon:
    """宝可梦类"""

    def __init__(self, name):
        self.name = name
        config = POKEMON_CONFIG[name]
        self.max_hp = config["HP"]
        self.hp = config["HP"]
        self.atk = config["ATK"]
        self.defense = config["DEF"]
        self.base_avd = config["AVD"]
        self.avd = config["AVD"]
        self.type = config["TYP"]
        self.max_left = config["LEFT"]
        self.left = config["LEFT"]
        self.buffs = []

    def is_fainted(self):
        """检查是否昏厥"""
        return self.hp <= 0

    def add_buff(self, buff_name, count=1):
        """添加状态"""
        for _ in range(count):
            self.buffs.append(buff_name)

    def remove_buff(self, buff_name, count=1):
        """移除状态"""
        removed = 0
        while buff_name in self.buffs and removed < count:
            self.buffs.remove(buff_name)
            removed += 1

    def has_buff(self, buff_name):
        """检查是否有某个状态"""
        return buff_name in self.buffs

    def get_buff_count(self, buff_name):
        """获取某个状态的层数"""
        return self.buffs.count(buff_name)

    def restore_health(self, amount):
        """回复生命值"""
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp

    def take_damage(self, damage):
        """受到伤害"""
        self.hp = max(0, self.hp - damage)
        return self.hp

    def reset_avd(self):
        """重置闪避率"""
        self.avd = self.base_avd

    def __str__(self):
        return f"{self.name}(HP: {self.hp}/{self.max_hp})"


class Battle:
    """对战类"""

    def __init__(self):
        self.player_team = []
        self.computer_team = []
        self.current_player = None
        self.current_computer = None
        self.turn_count = 0
        self.player_clean_used = False
        self.player_cure_used = False
        self.computer_clean_used = False
        self.computer_cure_used = False

    def setup_teams(self):
        """设置对战队伍"""
        # 电脑随机选择3只
        self.computer_team = [Pokemon(name) for name in random.sample(POKEMON_LIST, 3)]

        # 玩家选择3只
        print("请选择三只宝可梦组成队伍（输入编号，空格分开）：")
        for i, name in enumerate(POKEMON_LIST, 1):
            print(f"{i}.{name}", end="    ")
        print()

        while True:
            try:
                choices = list(map(int, input("你的选择：").split()))
                if len(choices) != 3 or len(set(choices)) != 3:
                    print("请选择3只不同的宝可梦！")
                    continue
                if any(choice < 1 or choice > len(POKEMON_LIST) for choice in choices):
                    print("请输入有效的编号！")
                    continue

                self.player_team = [Pokemon(POKEMON_LIST[choice - 1]) for choice in choices]
                break
            except (ValueError, IndexError):
                print("请输入有效的编号！")

    def choose_starting_pokemon(self):
        """选择首发宝可梦"""
        print("请选择出战的宝可梦：")
        for i, pokemon in enumerate(self.player_team, 1):
            print(f"{i}.{pokemon.name} (HP: {pokemon.hp}/{pokemon.max_hp})")

        while True:
            try:
                choice = int(input("你的选择："))
                if 1 <= choice <= len(self.player_team):
                    self.current_player = self.player_team[choice - 1]
                    break
                print("请输入有效的编号！")
            except ValueError:
                print("请输入数字！")

        self.current_computer = self.computer_team[0]
        print(f"你选择了 {self.current_player.name}!")
        print(f"电脑选择了 {self.current_computer.name}!")

    def calculate_damage(self, attacker, defender, atk_rate):
        """计算伤害"""
        base_damage = max(1, attacker.atk - defender.defense) * atk_rate

        # 属性克制计算
        attacker_type = attacker.type
        defender_type = defender.type

        # 克制关系: 1>2>3>4>1
        is_super_effective = ((attacker_type == 1 and defender_type == 2) or
                              (attacker_type == 2 and defender_type == 3) or
                              (attacker_type == 3 and defender_type == 4) or
                              (attacker_type == 4 and defender_type == 1))

        is_not_very_effective = ((defender_type == 1 and attacker_type == 2) or
                                 (defender_type == 2 and attacker_type == 3) or
                                 (defender_type == 3 and attacker_type == 4) or
                                 (defender_type == 4 and attacker_type == 1))

        if is_super_effective:
            base_damage *= 1.3
            print("效果拔群！")
        elif is_not_very_effective:
            base_damage *= 0.7
            print("效果一般...")

        # 感电效果
        if defender.has_buff("感电"):
            damage_multiplier = 1 + 0.1 * defender.get_buff_count("感电")
            base_damage *= damage_multiplier

        # 燃烧目标额外伤害（火属性）
        if attacker.type == 3 and defender.has_buff("燃烧"):
            if attacker.has_buff("引燃"):
                base_damage *= 1.3
            else:
                base_damage *= 1.15

        return round(max(1, base_damage), 2)

    def check_hit(self, defender):
        """检查是否命中"""
        if random.random() < defender.avd:
            # 触发闪避
            if defender.name == "杰尼龟" and defender.has_buff("水形"):
                # 水形闪避特效
                water_shapes = defender.get_buff_count("水形")
                heal_amount = water_shapes * 5
                defender.restore_health(heal_amount)
                defender.remove_buff("水形", water_shapes)
                defender.reset_avd()
                print(f"{defender.name} 闪避攻击并回复了{heal_amount}点HP！")
            else:
                print(f"{defender.name} 闪避了攻击！")
            return False
        return True

    def apply_type_effects(self, attacker, defender, skill_effect=False):
        """应用属性特效"""
        # 电属性特效
        if attacker.type == 1 and random.random() < 0.3:
            defender.add_buff("感电")
            print(f"{defender.name} 获得一层[感电]！")

        # 水属性受击特效（仅当被攻击时）
        if not skill_effect and defender.type == 2 and random.random() < 0.5:
            defender.add_buff("水形")
            defender.avd = min(0.9, defender.avd + 0.1)
            print(f"{defender.name} 获得一层[水形]！")

    def apply_turn_end_effects(self):
        """应用回合结束效果"""
        # 玩家宝可梦回合结束效果
        if self.current_player.has_buff("燃烧"):
            damage = 5
            self.current_player.take_damage(damage)
            self.current_player.remove_buff("燃烧", 1)
            print(f"{self.current_player.name} 因[燃烧]受到{damage}点伤害！")

        if self.current_player.has_buff("中毒"):
            poison_count = self.current_player.get_buff_count("中毒")
            damage = poison_count * 3
            self.current_player.take_damage(damage)
            print(f"{self.current_player.name} 因[中毒]受到{damage}点伤害！")

        # 草属性回合结束回复
        if self.current_player.type == 4:
            heal_amount = int(self.current_player.max_hp * 0.1)
            self.current_player.restore_health(heal_amount)
            print(f"{self.current_player.name} 的草属性特性回复了{heal_amount}点HP！")

            # 草属性净化效果
            if random.random() < 0.1:
                debuffs_removed = 0
                for buff in ["感电", "麻痹", "燃烧", "中毒"]:
                    if self.current_player.has_buff(buff):
                        self.current_player.remove_buff(buff)
                        debuffs_removed += 1
                if debuffs_removed > 0:
                    print(f"{self.current_player.name} 的草属性特性净化了负面状态！")

        # 电脑宝可梦回合结束效果
        if self.current_computer.has_buff("燃烧"):
            damage = 5
            self.current_computer.take_damage(damage)
            self.current_computer.remove_buff("燃烧", 1)
            print(f"{self.current_computer.name} 因[燃烧]受到{damage}点伤害！")

        if self.current_computer.has_buff("中毒"):
            poison_count = self.current_computer.get_buff_count("中毒")
            damage = poison_count * 3
            self.current_computer.take_damage(damage)
            print(f"{self.current_computer.name} 因[中毒]受到{damage}点伤害！")

        # 草属性回合结束回复
        if self.current_computer.type == 4:
            heal_amount = int(self.current_computer.max_hp * 0.1)
            self.current_computer.restore_health(heal_amount)
            print(f"{self.current_computer.name} 的草属性特性回复了{heal_amount}点HP！")

            # 草属性净化效果
            if random.random() < 0.1:
                debuffs_removed = 0
                for buff in ["感电", "麻痹", "燃烧", "中毒"]:
                    if self.current_computer.has_buff(buff):
                        self.current_computer.remove_buff(buff)
                        debuffs_removed += 1
                if debuffs_removed > 0:
                    print(f"{self.current_computer.name} 的草属性特性净化了负面状态！")

    def player_attack(self):
        """玩家攻击"""
        if self.current_player.name == "皮卡丘":
            self.pikachu_attack("player")
        elif self.current_player.name == "杰尼龟":
            self.jienigui_attack("player")
        elif self.current_player.name == "小火龙":
            self.xiaohuolong_attack("player")
        elif self.current_player.name == "妙蛙种子":
            self.miaowazhongzi_attack("player")
        elif self.current_player.name == "卡比兽":
            self.kabishou_attack("player")

    def pikachu_attack(self, attacker_type):
        """皮卡丘攻击"""
        attacker = self.current_player if attacker_type == "player" else self.current_computer
        defender = self.current_computer if attacker_type == "player" else self.current_player

        if attacker_type == "player":
            print(f"技能：1.电光一闪 2.十万伏特（剩余{attacker.left}次）")
            try:
                choice = int(input("选择技能："))
            except ValueError:
                choice = 1
        else:
            # 电脑AI：有感电层数时更可能用大招
            if defender.get_buff_count("感电") >= 2 and attacker.left > 0 and random.random() < 0.7:
                choice = 2
            else:
                choice = 1

        if choice == 1:
            print(f"{attacker.name} 使用了电光一闪！")
            # 第一击
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.0)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")
                self.apply_type_effects(attacker, defender)

            # 连击判定
            if random.random() < 0.2:
                print("触发了连击！")
                sleep(0.5)
                if self.check_hit(defender):
                    damage = self.calculate_damage(attacker, defender, 0.5)
                    defender.take_damage(damage)
                    print(f"连击造成{damage}点伤害！")
                    self.apply_type_effects(attacker, defender)

        elif choice == 2 and attacker.left > 0:
            attacker.left -= 1
            print(f"{attacker.name} 使用了十万伏特！")
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.6)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")

                # 麻痹判定
                paralyze_chance = 0.3 + 0.2 * defender.get_buff_count("感电")
                if random.random() < paralyze_chance:
                    defender.add_buff("麻痹")
                    print(f"{defender.name} 被麻痹了！")

                self.apply_type_effects(attacker, defender)
        else:
            print("技能次数不足，使用电光一闪！")
            self.pikachu_attack(attacker_type)

    def jienigui_attack(self, attacker_type):
        """杰尼龟攻击"""
        attacker = self.current_player if attacker_type == "player" else self.current_computer
        defender = self.current_computer if attacker_type == "player" else self.current_player

        if attacker_type == "player":
            print(f"技能：1.水枪 2.水之波动（剩余{attacker.left}次）")
            try:
                choice = int(input("选择技能："))
            except ValueError:
                choice = 1
        else:
            # 电脑AI：血量低时更可能用大招
            if attacker.hp < attacker.max_hp * 0.5 and attacker.left > 0 and random.random() < 0.6:
                choice = 2
            else:
                choice = 1

        if choice == 1:
            print(f"{attacker.name} 使用了水枪！")
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.1)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")
                self.apply_type_effects(attacker, defender)

            # 水形判定
            if random.random() < 0.15:
                attacker.add_buff("水形")
                attacker.avd = min(0.9, attacker.avd + 0.1)
                print(f"{attacker.name} 获得一层[水形]！")

        elif choice == 2 and attacker.left > 0:
            attacker.left -= 1
            print(f"{attacker.name} 使用了水之波动！")
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.6)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")

            attacker.add_buff("水形")
            attacker.avd = min(0.9, attacker.avd + 0.1)
            print(f"{attacker.name} 获得一层[水形]！")
            self.apply_type_effects(attacker, defender)
        else:
            print("技能次数不足，使用水枪！")
            self.jienigui_attack(attacker_type)

    def xiaohuolong_attack(self, attacker_type):
        """小火龙攻击"""
        attacker = self.current_player if attacker_type == "player" else self.current_computer
        defender = self.current_computer if attacker_type == "player" else self.current_player

        if attacker_type == "player":
            print(f"技能：1.火花 2.龙之怒（剩余{attacker.left}次）")
            try:
                choice = int(input("选择技能："))
            except ValueError:
                choice = 1
        else:
            if random.random() < 0.5 and attacker.left > 0:
                choice = 2
            else:
                choice = 1

        if choice == 1:
            print(f"{attacker.name} 使用了火花！")
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.2)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")

                # 燃烧判定
                if random.random() < 0.2 or attacker.has_buff("引燃"):
                    defender.add_buff("燃烧", 2)
                    print(f"{defender.name} 被点燃了！")

                self.apply_type_effects(attacker, defender)

        elif choice == 2 and attacker.left > 0:
            attacker.left -= 1
            print(f"{attacker.name} 使用了龙之怒！")
            attacker.add_buff("引燃", 2)
            print(f"{attacker.name} 获得[引燃]效果！")

            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 1.6)
                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")

                defender.add_buff("燃烧", 2)
                print(f"{defender.name} 被点燃了！")

                self.apply_type_effects(attacker, defender)
        else:
            print("技能次数不足，使用火花！")
            self.xiaohuolong_attack(attacker_type)

    def miaowazhongzi_attack(self, attacker_type):
        """妙蛙种子攻击"""
        attacker = self.current_player if attacker_type == "player" else self.current_computer
        defender = self.current_computer if attacker_type == "player" else self.current_player

        if attacker_type == "player":
            print(f"技能：1.毒粉 2.寄生种子（剩余{attacker.left}次）")
            try:
                choice = int(input("选择技能："))
            except ValueError:
                choice = 1
        else:
            if random.random() < 0.8 and attacker.left > 0:
                choice = 2
            else:
                choice = 1

        if choice == 1:
            print(f"{attacker.name} 使用了毒粉！")
            if self.check_hit(defender):
                damage = self.calculate_damage(attacker, defender, 0.7)
                # 百毒不侵减伤
                if defender.has_buff("百毒不侵") and attacker.has_buff("中毒"):
                    damage *= 0.8

                defender.take_damage(damage)
                print(f"造成{damage}点伤害！")

                defender.add_buff("中毒")
                print(f"{defender.name} 中毒了！")

                self.apply_type_effects(attacker, defender)

        elif choice == 2 and attacker.left > 0:
            attacker.left -= 1
            print(f"{attacker.name} 使用了寄生种子！")

            defender.add_buff("中毒", 2)
            attacker.add_buff("百毒不侵", 3)

            print(f"{defender.name} 中了剧毒！")
            print(f"{attacker.name} 获得[百毒不侵]效果！")
        else:
            print("技能次数不足，使用毒粉！")
            self.miaowazhongzi_attack(attacker_type)

    def kabishou_attack(self, attacker_type):
        """卡比兽攻击"""
        attacker = self.current_player if attacker_type == "player" else self.current_computer
        defender = self.current_computer if attacker_type == "player" else self.current_player

        if attacker_type == "player":
            print(f"技能：1.撞击 2.百万吨重拳（剩余{attacker.left}次）")
            try:
                choice = int(input("选择技能："))
            except ValueError:
                choice = 1
        else:
            if attacker.left > 0:
                choice = 2
            else:
                choice = 1

        # 卡比兽技能必中
        if choice == 1:
            print(f"{attacker.name} 使用了撞击！")
            damage = self.calculate_damage(attacker, defender, 1.4)
            defender.take_damage(damage)
            print(f"造成{damage}点伤害！")

            # 眩晕判定
            if random.random() < 0.3:
                defender.add_buff("眩晕")
                print(f"{defender.name} 被眩晕了！")

            self.apply_type_effects(attacker, defender, True)

        elif choice == 2 and attacker.left > 0:
            attacker.left -= 1
            print(f"{attacker.name} 使用了百万吨重拳！")
            damage = self.calculate_damage(attacker, defender, 1.8)
            defender.take_damage(damage)
            print(f"造成{damage}点伤害！")

            # 眩晕判定
            if random.random() < 0.5:
                defender.add_buff("眩晕")
                print(f"{defender.name} 被眩晕了！")

            self.apply_type_effects(attacker, defender, True)
        else:
            print("技能次数不足，使用撞击！")
            self.kabishou_attack(attacker_type)

    def player_cleanse(self):
        """玩家净化"""
        if not self.player_clean_used:
            debuffs_removed = 0
            for buff in ["感电", "麻痹", "燃烧", "中毒", "眩晕"]:
                count = self.current_player.get_buff_count(buff)
                if count > 0:
                    self.current_player.remove_buff(buff, count)
                    debuffs_removed += count

            if debuffs_removed > 0:
                print("净化成功！所有负面状态被清除！")
            else:
                print("没有负面状态需要净化！")

            self.player_clean_used = True
        else:
            print("净化次数已用完！")

    def player_heal(self):
        """玩家回复"""
        if not self.player_cure_used:
            old_hp = self.current_player.hp
            self.current_player.restore_health(30)
            heal_amount = self.current_player.hp - old_hp
            print(f"{self.current_player.name} 回复了{heal_amount}点HP！")
            print(f"当前HP: {self.current_player.hp}/{self.current_player.max_hp}")
            self.player_cure_used = True
        else:
            print("回复次数已用完！")

    def player_switch(self):
        """玩家替换宝可梦"""
        available_pokemon = [p for p in self.player_team if not p.is_fainted() and p != self.current_player]

        if not available_pokemon:
            print("没有可替换的宝可梦！")
            return True

        print("选择替换的宝可梦：")
        for i, pokemon in enumerate(available_pokemon, 1):
            print(f"{i}.{pokemon.name} (HP: {pokemon.hp}/{pokemon.max_hp})")

        try:
            choice = int(input("你的选择："))
            if 1 <= choice <= len(available_pokemon):
                new_pokemon = available_pokemon[choice - 1]
                print(f"换下 {self.current_player.name}，换上 {new_pokemon.name}！")
                self.current_player = new_pokemon
                return True
            else:
                print("无效选择！")
                return self.player_switch()
        except ValueError:
            print("请输入数字！")
            return self.player_switch()

    def computer_attack(self):
        """电脑攻击"""
        if self.current_computer.name == "皮卡丘":
            self.pikachu_attack("computer")
        elif self.current_computer.name == "杰尼龟":
            self.jienigui_attack("computer")
        elif self.current_computer.name == "小火龙":
            self.xiaohuolong_attack("computer")
        elif self.current_computer.name == "妙蛙种子":
            self.miaowazhongzi_attack("computer")
        elif self.current_computer.name == "卡比兽":
            self.kabishou_attack("computer")

    def computer_cleanse(self):
        """电脑净化"""
        if not self.computer_clean_used and len(
                [b for b in self.current_computer.buffs if b in ["感电", "麻痹", "燃烧", "中毒", "眩晕"]]) >= 3:
            for buff in ["感电", "麻痹", "燃烧", "中毒", "眩晕"]:
                self.current_computer.remove_buff(buff)
            print(f"{self.current_computer.name} 清除了所有负面状态！")
            self.computer_clean_used = True

    def computer_heal(self):
        """电脑回复"""
        if not self.computer_cure_used and self.current_computer.hp < self.current_computer.max_hp * 0.4:
            old_hp = self.current_computer.hp
            self.current_computer.restore_health(30)
            heal_amount = self.current_computer.hp - old_hp
            print(f"{self.current_computer.name} 回复了{heal_amount}点HP！")
            self.computer_cure_used = True

    def computer_switch(self):
        """电脑替换宝可梦"""
        available_pokemon = [p for p in self.computer_team if not p.is_fainted() and p != self.current_computer]

        if available_pokemon:
            new_pokemon = random.choice(available_pokemon)
            print(f"电脑换下 {self.current_computer.name}，换上 {new_pokemon.name}！")
            self.current_computer = new_pokemon

    def execute_player_turn(self):
        """执行玩家回合"""
        print("\n-----你的回合-----")

        if self.current_player.is_fainted():
            print("当前宝可梦已昏厥！")
            self.current_player.add_buff("昏厥")
            return self.player_switch()

        if self.current_player.has_buff("眩晕"):
            print("当前宝可梦陷入眩晕！无法行动！！")
            self.current_player.remove_buff("眩晕")
            return True

        if self.current_player.has_buff("麻痹"):
            print("当前宝可梦被麻痹！无法攻击！")
            self.current_player.remove_buff("麻痹")
            options = ["2", "3", "4", "5", "6"]
        else:
            options = ["1", "2", "3", "4", "5", "6"]

        print("选择行动: 1.攻击 2.净化 3.回复 4.替换 5.空过 6.逃跑")
        choice = input("你的选择：")

        if choice not in options:
            print("无效选择，自动空过")
            return True

        if choice == "1":
            self.player_attack()
        elif choice == "2":
            self.player_cleanse()
        elif choice == "3":
            self.player_heal()
        elif choice == "4":
            return self.player_switch()
        elif choice == "5":
            print("空过一回合")
        elif choice == "6":
            print("玩家逃跑！战斗失败！")
            return False

        return True

    def execute_computer_turn(self):
        """执行电脑回合"""
        print("\n-----电脑的回合-----")

        if self.current_computer.is_fainted():
            print(f"{self.current_computer.name} 已昏厥！")
            self.current_computer.add_buff("昏厥")
            self.computer_switch()
            return

        if self.current_computer.has_buff("眩晕"):
            print(f"{self.current_computer.name} 被眩晕，无法行动！")
            self.current_computer.remove_buff("眩晕")
            return

        # 电脑决策
        if self.current_computer.has_buff("麻痹"):
            print(f"{self.current_computer.name} 被麻痹了！")
            self.current_computer.remove_buff("麻痹")

            if self.current_computer.hp < self.current_computer.max_hp * 0.5:
                self.computer_heal()
            else:
                print("电脑选择空过")
        else:
            # 先检查是否需要净化或回复
            self.computer_cleanse()
            self.computer_heal()

            # 然后攻击
            self.computer_attack()

    def check_winner(self):
        """检查胜负"""
        player_fainted = all(pokemon.is_fainted() for pokemon in self.player_team)
        computer_fainted = all(pokemon.is_fainted() for pokemon in self.computer_team)

        if player_fainted:
            print("\n所有宝可梦都昏厥了！电脑获胜！")
            return True
        elif computer_fainted:
            print("\n电脑的所有宝可梦都昏厥了！你获胜！")
            return True

        return False

    def declare_timeout_winner(self):
        """20回合结束时宣布胜负"""
        player_total_hp = sum(pokemon.hp for pokemon in self.player_team)
        computer_total_hp = sum(pokemon.hp for pokemon in self.computer_team)

        print("\n20回合结束！")
        print(f"你的队伍总HP: {player_total_hp}")
        print(f"电脑队伍总HP: {computer_total_hp}")

        if player_total_hp > computer_total_hp:
            print("你获胜！")
        elif computer_total_hp > player_total_hp:
            print("电脑获胜！")
        else:
            print("平局！")

    def display_status(self):
        """显示当前状态"""
        print(f"\n你的宝可梦: {self.current_player}")
        print(f"电脑宝可梦: {self.current_computer}")

        player_buffs = set(self.current_player.buffs)
        computer_buffs = set(self.current_computer.buffs)

        if player_buffs:
            print(f"你的状态: {', '.join(player_buffs)}")
        if computer_buffs:
            print(f"电脑状态: {', '.join(computer_buffs)}")

    def run(self):
        """运行对战"""
        self.setup_teams()
        self.choose_starting_pokemon()

        for turn in range(1, 21):
            print(f"\n{'=' * 20} 第{turn}回合 {'=' * 20}")
            self.display_status()

            # 玩家回合
            if not self.execute_player_turn():
                break

            # 检查胜负
            if self.check_winner():
                break

            # 电脑回合
            self.execute_computer_turn()

            # 检查胜负
            if self.check_winner():
                break

            # 回合结束效果
            self.apply_turn_end_effects()

            sleep(1)

        else:
            # 20回合结束
            self.declare_timeout_winner()


# 游戏介绍函数
class Intro:
    """游戏介绍类"""

    def rule(self):
        print("----------规则介绍----------")
        sleep(1)
        printer("每次战斗前，玩家可以选择三只宝可梦组成队伍，与电脑对战（电脑随机选择宝可梦），对战开始时选择一个宝可梦出战")
        sleep(1)
        printer("由玩家先开始，玩家与电脑轮流行动，每回合可以从【攻击】、【净化】、【回复】、【替换】、【逃跑】中选择一项")
        sleep(1)
        printer("每只宝可梦都有自己的属性、属性值以及技能，每个属性也有不同的特性")
        sleep(1)
        printer("当宝可梦的生命值小于等于0时，将陷入[昏厥]，无法进行【攻击】、【净化】与【恢复】")
        sleep(1)
        printer("当一方的所有宝可梦都陷入[昏厥]后，视为其战斗失败")
        sleep(1)
        printer("最多进行20回合，20回合结束后，剩余宝可梦生命值总和最高的一方获胜")
        sleep(1)
        printer("【攻击】：从宝可梦技能中选择一项对目标进行攻击\n【回复】：（每场战斗（所有宝可梦）限1次），回复当前宝可梦30点生命值\n【净化】：（每场战斗限1次），清除当前宝可梦所有减益状态\n【替换】：从未[昏厥]的宝可梦中选择一只替换场上宝可梦，保留所有属性、状态\n【逃跑】：直接视为战斗失败")


    def type(self):
        printer("共有电、水、火、草四个属性")
        sleep(1)
        printer("其中：电 克制 水\t水 克制 火\t火 克制 草\t草 克制 电")
        sleep(1)
        printer("目标属性被自身克制时，攻击额外造成30%伤害；目标属性克制自身时，攻击减少30%伤害\n")
        sleep(1)
        printer("电属性特性：攻击命中后有30%概率永久赋予目标一层[感电]\n[感电]：受到伤害提升10%，最多叠加3层\n")
        sleep(1)
        printer("水属性特性：受击后50%赋予自身一层[水形]\n[水形]：使自身闪避率提高10%，可无限叠加，成功闪避攻击后清空层数，并回复自身[水形]层数*5点生命值\n")
        sleep(1)
        printer("火属性特性:攻击命中后有20%概率赋予目标[燃烧]2回合，火属性宝可梦对存在[燃烧]的目标造成的伤害额外提升15%\n[燃烧]：回合结束后受到5点伤害\n")
        sleep(1)
        printer("草属性特性：回合结束后回复自身10%最大生命值,且有10%概率净化自身状态减益")


    def buff(self):
        print("----------状态减益----------")
        sleep(0.5)
        printer("[昏厥]：无法进行【攻击】、【恢复】至本场战斗结束；若一名玩家所有宝可梦都陷入[昏厥]，则战斗失败\n[感电]：受到伤害提升10%，最多叠加3层\n[麻痹]:无法进行【攻击】\n[燃烧]：回合结束后受到3点伤害\n[中毒]：可无限叠加，回合结束受到[中毒]层数*5点伤害\n[眩晕]：无法行动")
        print()
        sleep(1)
        printer("----------状态增益----------")
        sleep(0.5)
        printer("[水形]：使自身闪避率提高10%，可无限叠加，成功闪避攻击后清空层数，并回复自身[水形]层数*5点生命值\n[引燃]：自身攻击命中后，赋予目标[燃烧]3回合；且自身对携带[燃烧]的目标造成的额外伤害提升至30%\n[百毒不侵]：受到拥有[中毒]的敌人的伤害减少20%")


    def pokemon(self,name):
        global HP, ATK, DEF, AVD, TYP
        if name == "皮卡丘":
            print("皮卡丘：电属性宝可梦\n")
            sleep(1)
            print(f"生命值：{HP["皮卡丘"]}  攻击力：{ATK["皮卡丘"]}  防御力：{DEF["皮卡丘"]}  闪避率：{AVD["皮卡丘"]}\n")
            sleep(1)
            print("技能一：电光一闪\n对目标造成100%攻击力电属性伤害，有20%概率连击一次，造成50%攻击力电属性伤害\n")
            sleep(1)
            print("技能二：十万伏特\n（每场战斗限1次）对目标造成160%攻击力电属性伤害；命中后有30%使目标[麻痹]一回合；目标每有一层[感电]，额外有20%概率使目标[麻痹]\n[麻痹]：无法进行【攻击】")
        elif name == "杰尼龟":
            print("杰尼龟：水属性宝可梦\n")
            sleep(1)
            print(f"生命值：{HP["杰尼龟"]}  攻击力：{ATK["杰尼龟"]}  防御力：{DEF["杰尼龟"]}  闪避率：{AVD["杰尼龟"]}\n")
            sleep(1)
            print("技能一：水枪\n对目标造成110%攻击力水属性伤害，攻击后有15%概率赋予自身一层[水形]\n")
            sleep(1)
            print("技能二：水之波动\n（每场战斗限2次）对目标造成160%攻击力水属性伤害，攻击后赋予自身一层[水形]")
        elif name == "小火龙":
            print("小火龙：火属性宝可梦\n")
            sleep(1)
            print(f"生命值：{HP["小火龙"]}  攻击力：{ATK["小火龙"]}  防御力：{DEF["小火龙"]}  闪避率：{AVD["小火龙"]}\n")
            sleep(1)
            print("技能一：火花\n对目标造成120%攻击力火属性伤害\n")
            sleep(1)
            print("技能二：龙之怒\n（每场战斗限一次）使自身获得[引燃]2回合，然后对目标造成160%攻击力伤害,命中后赋予目标[燃烧]2回合\n[引燃]：自身对携带[燃烧]的目标造成的额外伤害提升至30%")
        elif name == "妙蛙种子":
            print("妙蛙种子：草属性宝可梦\n")
            sleep(1)
            print(f"生命值：{HP["妙蛙种子"]}  攻击力：{ATK["妙蛙种子"]}  防御力：{DEF["妙蛙种子"]}  闪避率：{AVD["妙蛙种子"]}\n")
            sleep(1)
            print("技能一：毒粉\n对目标造成70%攻击力草属性伤害，命中后赋予目标一层[中毒]\n[中毒]：可无限叠加，回合结束受到[中毒]层数*3点伤害\n")
            sleep(1)
            print("技能二：寄生种子\n（每场战斗限1次）赋予目标2层[中毒]；赋予自身[百毒不侵]3回合\n[百毒不侵]：自身受到拥有[中毒]敌人的伤害降低20%")
        elif name == "卡比兽":
            print("卡比兽：火属性宝可梦\n")
            sleep(1)
            print(f"生命值：{HP["卡比兽"]}  攻击力：{ATK["卡比兽"]}  防御力：{DEF["卡比兽"]}  闪避率：{AVD["卡比兽"]}\n")
            sleep(1)
            print("技能一：撞击\n必中，对目标造成140%攻击力火属性伤害，有30%概率使目标[眩晕]一回合\n")
            sleep(1)
            print("技能二：百万吨重拳\n（每场战斗限两次）必中，对目标造成180%攻击力火属性伤害，有50%概率使目标[眩晕]一回合\n[眩晕]：无法行动")


    def game(self):
        print("-----------游戏介绍----------")
        print("规则介绍\t\t[输入1]\n属性介绍\t\t[输入2]\n状态介绍\t\t[输入3]\n宝可梦图鉴\t[输入4]\n其他任意键返回上一页")
        choice_intro = input("你的选择：")

        if choice_intro == "1":
            self.rule()
            print()
            n = input("任意键返回上一页")
            self.game()
        elif choice_intro == "2":
            self.type()
            print()
            n = input("任意键返回上一页")
            self.game()
        elif choice_intro == "3":
            self.buff()
            print()
            n = input("任意键返回上一页")
            self.game()
        elif choice_intro == "4":
            self.menu_pokemon()
            print()
            n = input("任意键返回上一页")
            self.game()
        else:
            main()


    def menu_pokemon(self):
        print("------宝可梦图鉴-----")
        print("皮卡丘\t\t[输入1]\n杰尼龟\t\t[输入2]\n小火龙\t\t[输入3]\n妙蛙种子\t\t[输入4]\n卡比兽\t\t[输入5]\n其他任意键返回上一页")
        choice_pokemon = input("你的选择：")
        if choice_pokemon == "1":
            self.pokemon("皮卡丘")
            print()
            n = input("任意键返回上一页")
            self.menu_pokemon()
        elif choice_pokemon == "2":
            self.pokemon("杰尼龟")
            print()
            n = input("任意键返回上一页")
            self.menu_pokemon()
        elif choice_pokemon == "3":
            self.pokemon("小火龙")
            print()
            n = input("任意键返回上一页")
            self.menu_pokemon()
        elif choice_pokemon == "4":
            self.pokemon("妙蛙种子")
            print()
            n = input("任意键返回上一页")
            self.menu_pokemon()
        elif choice_pokemon == "5":
            self.pokemon("卡比兽")
            print()
            n = input("任意键返回上一页")
            self.menu_pokemon()
        else:
            self.game()


def main():
    """主函数"""
    while True:
        print("\n----------宝可梦对战----------")
        print("1.开始游戏\n2.游戏介绍\n3.退出游戏")

        choice = input("你的选择：")

        if choice == "1":
            battle = Battle()
            battle.run()
        elif choice == "2":
            intro = Intro()
            intro.game()
        elif choice == "3":
            print("再见！")
            break
        else:
            print("无效选择！")


if __name__ == "__main__":
    main()