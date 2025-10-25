import random
from time import sleep


def Introduce():
    print("""欢迎来到宝可梦的世界，准备好开启一场酣畅淋漓的对战了吗？\t
    玩家可以选择三个宝可梦组成队伍与电脑进行对战\t
    现在游戏中有五种属性，克制关系：水——→草——→火——→电——→光——→水\t
    尽情玩耍吧！\t""")
    character_map = input(
        "若要查看角色图鉴 以及 技能效果，请输入:‘map’ \n任意输入可直接开始"
    )
    # 确保变量互不相等且都在字典中
    if character_map == "map":
        print("""
        属性被动效果：\t
        1.水属性被动：每回合50%概率，减免30%受到的伤害\t
        2.草属性被动：每回合回复10%最大生命值血量\t
        3.火属性被动：对对手造成伤害时，提高10%初始攻击力，最高叠加4层\t
        4.电属性被动：闪避成功后立即发动一次技能\t
        5.光属性被动：死亡后可以复活一次并回复50%的最大生命值""")
        print("""角色图鉴：\t
        一.水属性角色：\t
            1.杰尼龟：\t
              技能1：水枪： *杰尼龟喷射出一股强力的水流，对敌方造成140%水属性伤害\t
              技能2：护盾： *杰尼龟使用水流形成保护盾，减少50%下一次受到的伤害，可叠加\t
        二.草属性角色：\t
            1.妙蛙种子：\t
              技能1：种子炸弹： *妙蛙种子发射一颗种子，爆炸后对敌方造成1.0倍攻击力伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失10%生命值,持续2回合\t
              技能2：寄生种子： *妙蛙种子向对手播种，每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合\t
        三.火属性角色：\t
            1.小火龙：\t
              技能1：火花： *小火龙发射出一团小火焰，对敌人造成 100% 火属性伤害，并有10%的几率使目标陷入“烧伤”状态，每回合受到10额外伤害， 层数不限\t
              技能2：蓄能爆炎： *小火龙召唤出强大的火焰，对敌人造成 300% 火属性伤害，并有80%的几率使敌人陷入“烧伤”状态，这个技能需要1个回合的蓄力\t
                    敌方在面对改技能时闪避率增加 20%\t
        四.电属性角色：\t
            1.皮卡丘：\t
              技能1：十万伏特： *对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人麻痹，使敌人跳过1回合\t
              技能2：电光一闪： *对敌人造成1.0倍攻击力的快速攻击，有30%触发第二次攻击\t
        五.光属性角色：\t
            1.玛卡巴卡：\t
              角色被动：玛卡巴卡受到最大伤害不超过5点，且血量小于等于5点时，闪避率提高0.3，死亡复活后进入二阶段\t           
              技能1：拍手：*回复自身2点生命值 并提高自身5点攻击力(复活后仍继承攻击力)\t
              技能2：一阶段（复活前）：玛卡巴卡：*造成100%攻击力的光属性伤害\t
                    二阶段（复活后）：超级玛卡巴卡：*扣除自身5点生命值(最低扣至0.01)，造成200%攻击力的光属性伤害，有30%几率再次发动技能\t
                    有1%的概率使出技能 玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡.....造成999点伤害\t
                    """)
        print(input("若准备好，请输入回车以开始游戏"))
        print("游戏开始！！！")
    else:
        print("游戏开始！！！")


# Class模块
class Pokemon:
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        self.name = name
        self.HP = HP
        self.max_HP = max_HP
        self.ATK = ATK
        self.initial_ATK = initial_ATK
        self.DEF = DEF
        self.property = property
        self.dodge_probability = dodge_probability
        self.status = status
        self.dodge_judgement = False
        self.flame_charge_status = None
        self.flame_charge2_status = None
        self.status_list = []
        self.skip_judge = False
        self.burning_count = 0

    def dodge(self):
        if random.random() < float(self.dodge_probability):
            return True
        else:
            return False

    def restrain(self, opponent):  # 置于每次技能damage计算出之后
        # 克制关系()：水——→草——→火——→电——→光——→水
        restrain_list = [
            ("水", "草"),
            ("草", "火"),
            ("火", "电"),
            ("电", "光"),
            ("光", "水"),
        ]
        if (self.property, opponent.property) in restrain_list:
            print(f"由于属性克制,{self.name}的伤害降低50%")
            sleep(1.6)
            self.damage = self.damage / 2
            return self.damage
        elif (opponent.property, self.property) in restrain_list:
            print(f"由于属性克制,{self.name}的伤害翻倍")
            sleep(1.6)
            self.damage = self.damage * 2
            return self.damage
        else:
            return None

    def print_restrain(self, opponent):  # 置于选择角色后
        # 克制关系()：水——→草——→火——→电——→光——→水
        restrain_list = [
            ("水", "草"),
            ("草", "火"),
            ("火", "电"),
            ("电", "光"),
            ("光", "水"),
        ]
        if (self.property, opponent.property) in restrain_list:
            print(
                f"{self.name}的属性[{self.property}]被{opponent.name}的属性[{opponent.property}]克制\n"
                f"{self.name}对{opponent.name}造成伤害降低50%，受到{opponent.name}伤害翻倍"
            )
            sleep(2)
        elif (opponent.property, self.property) in restrain_list:
            print(
                f"{opponent.name}的属性[{opponent.property}]被{self.name}的属性[{self.property}]克制\n"
                f"{opponent.name}对{self.name}造成伤害降低50%，受到{self.name}伤害翻倍"
            )
            sleep(2)
        else:
            print("两个宝可梦间无克制关系")
            sleep(1)
        print()


class WaterPokemon(Pokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )

    def water_passive(self, opponent):
        damage_reduce_probability = 0.5
        if random.random() < float(damage_reduce_probability):
            opponent.damage = opponent.damage * 0.7
            print(f"{self.name} 成功发动水属性被动（50%概率），减免30%受到的伤害")
            sleep(1)


class GrassPokemon(Pokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )

    def grass_passive(self):
        if self.HP < self.max_HP:
            self.HP += self.max_HP * 0.1
            if self.HP >= self.max_HP:
                self.HP = self.max_HP
            print(
                f"{self.name} 发动草属性被动，回复 {self.max_HP * 0.1} 点（10%最大生命值）血量, 当前 {self.name} 的生命值为 {self.HP}"
            )
            print()
            sleep(2)


class FirePokemon(Pokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.fire_count = 0

    def fire_passive(self, opponent):  # dodge为上面定义的函数
        if opponent.dodge_judgement == False and self.damage > 0:
            while self.fire_count < 4:
                self.fire_count += 1
                self.ATK += 0.1 * self.initial_ATK
                print(
                    f"{self.name} 发动火属性被动，提高 {0.1 * self.initial_ATK} 点攻击力(10%初始攻击力，最高叠加4层)"
                )
                break
            print(
                f"{self.name}当前已叠加{self.fire_count}层被动，提高{0.1 * self.initial_ATK * self.fire_count}点攻击力"
            )
            sleep(1)


class ElectricalPokemon(Pokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )

    def user_electrical_passive(self, opponent):
        if self.dodge_judgement == True:
            print(f"{self.name} 成功闪避，触发电属性被动，立即发动一次技能")
            sleep(1)
            print()
            # 插入我方回合
            UserTerm(self, opponent)
            self.dodge_judgement = False

    def computer_electrical_passive(self, opponent):
        if self.dodge_judgement == True:
            print(f"{self.name} 成功闪避，触发电属性被动，立即发动一次技能")
            sleep(1.5)
            print()
            # 插入电脑
            ComputerTerm(opponent, self)
            self.dodge_judgement = False


class LightPokemon(Pokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.count_of_reactivation = 1
        self.max_HP = self.HP

    def light_passive(self):
        if self.count_of_reactivation == 1 and self.HP <= 0:
            print(f"{self.name} 触发光属性被动，复活并回复50%的最大生命值")
            sleep(2)
            self.HP = self.max_HP / 2
            self.count_of_reactivation = 0
            return self.HP, self.count_of_reactivation


# 宝可梦类定义
class Pikachu(ElectricalPokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.skill1_str = "十万伏特： *对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人麻痹(使敌人跳过1回合)"
        self.skill2_str = (
            "电光一闪： *对敌人造成1.0倍攻击力的快速攻击，有30%触发第二次攻击"
        )
        self.skill1 = self.thunderbolt  # 设置技能指向
        self.skill2 = self.quick_attack  # 设置技能指向

    def thunderbolt(self, opponent):
        print(f"{self.name} 使用了 十万伏特")
        sleep(1)
        thunderbolt_ATK = 1.4 * self.ATK
        thunderbolt_damage = thunderbolt_ATK - opponent.DEF
        self.damage = thunderbolt_damage
        if random.random() < 0.1:
            if opponent.dodge_judgement == False:
                print(f"雷电降临！！！（10%概率） {self.name} 对对手施加麻痹状态")
                sleep(1)
                opponent.status_list.append("Palsy")
                opponent.palsy_count = 1

    def quick_attack(self, opponent):
        print(f"{self.name} 使用了 电光一闪")
        sleep(1)
        quick_attack_ATK = self.ATK
        if random.random() < 0.3:
            print(
                f"{self.name} 成功召唤雷电（30%概率），再次使用了 电光一闪 ，造成双倍伤害"
            )
            sleep(1)
            quick_attack_damage = (quick_attack_ATK - opponent.DEF) * 2
        else:
            quick_attack_damage = quick_attack_ATK - opponent.DEF
        self.damage = quick_attack_damage


class Bulbasaur(GrassPokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.skill1_str = "种子炸弹： *妙蛙种子发射一颗种子，爆炸后对敌方造成1.0倍攻击力伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失10%生命值,持续2回合"
        self.skill2_str = "寄生种子： *妙蛙种子向对手播种，每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合"
        self.skill1 = self.seed_bomb  # 设置技能指向
        self.skill2 = self.parasitic_seeds  # 设置技能指向

    def seed_bomb(self, opponent):
        print(f"{self.name} 使用了 种子炸弹")
        sleep(1)
        seed_bomb_ATK = self.ATK
        seed_bomb_damage = seed_bomb_ATK - opponent.DEF
        self.damage = seed_bomb_damage
        if random.random() < 0.15:
            if opponent.dodge_judgement == False:
                print(f"{self.name} 对对手施加 “中毒” 状态")
                sleep(1)
                opponent.status_list.append("Poisoned")
                opponent.poisoned_count = 2

    def parasitic_seeds(self, opponent):
        self.damage = 0
        print(f"{self.name} 使用了 寄生种子")
        sleep(1)
        if opponent.dodge_judgement == False:
            self.status_list.append("ParasiticOpponent")
            self.parasitic_count = 3


class Squirtle(WaterPokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.skill1_str = "水枪： *杰尼龟喷射出一股强力的水流，对敌方造成140%水属性伤害"
        self.skill2_str = (
            "护盾： *杰尼龟使用水流形成保护盾，减少50%下一次受到的伤害，可叠加"
        )
        self.skill1 = self.aqua_jet  # 设置技能指向
        self.skill2 = self.shield  # 设置技能指向

    def aqua_jet(self, opponent):
        print(f"{self.name} 使用了 水枪")
        sleep(1)
        aqua_jet_ATK = 1.4 * self.ATK
        aqua_jet_damage = aqua_jet_ATK - opponent.DEF
        self.damage = aqua_jet_damage

    def shield(self, opponent):
        self.damage = 0
        print(f"{self.name} 使用了 护盾")
        sleep(1)
        opponent.name = opponent.name
        self.status_list.append("SquirtleShield")


class Charmander(FirePokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.skill1_str = "火花： *小火龙发射出一团小火焰，对敌人造成 100% 火属性伤害，并有10%的几率使目标陷入“烧伤”状态（每回合受到10额外伤害， 层数可无限叠加）"
        self.skill2_str = "蓄能爆炎： *小火龙召唤出强大的火焰，对敌人造成 300% 火属性伤害，并有80%的几率使敌人陷入“烧伤”状态，这个技能需要1个回合的蓄力，并且在面对改技能时敌方闪避率增加 20%"
        self.skill1 = self.ember  # 设置技能指向
        self.skill2 = self.flame_charge  # 设置技能指向

    def ember(self, opponent):
        print(f"{self.name} 使用了 火花")
        sleep(1)
        ember_ATK = self.ATK
        ember_damage = ember_ATK - opponent.DEF
        self.damage = ember_damage
        if random.random() < 0.1:
            if opponent.dodge_judgement == False:
                print(f"火神庇佑！(10%概率) {self.name} 对对手施加2回合“烧伤”状态")
                sleep(1)
                opponent.status_list.append("Burning")
                opponent.burning_count += 2

    def flame_charge(self, opponent):
        flame_charge_damage = 0
        if self.flame_charge_status == "Preparation":
            print(f"{self.name} 蓄力完成 发动技能 蓄能爆炎")
            sleep(1)
            flame_charge_damage = 3 * (self.ATK - opponent.DEF)
            if random.random() < 0.8:
                if opponent.dodge_judgement == False:
                    print(f"火神祝福！(80%概率) {self.name} 对对手施加2回合“烧伤”状态")
                    sleep(1)
                    opponent.status_list.append("Burning")
                    opponent.burning_count += 2
        if self.flame_charge_status is None:
            print(f"{self.name} 进行 蓄能爆炎 的蓄力")
            sleep(1)
            self.flame_charge_status = "Preparation"
        elif self.flame_charge_status == "Preparation":
            self.flame_charge_status = None
        self.damage = flame_charge_damage


class Makabaka(LightPokemon):
    def __init__(
        self,
        name,
        HP,
        max_HP,
        ATK,
        initial_ATK,
        DEF,
        property,
        dodge_probability,
        status,
    ):
        super().__init__(
            name, HP, max_HP, ATK, initial_ATK, DEF, property, dodge_probability, status
        )
        self.skill1_str = "拍手：*回复自身2点生命值 并提高自身5点攻击力"
        self.skill2_str = (
            "一阶段（复活前）：玛卡巴卡：*造成100%攻击力的光属性伤害\t\n     "
            "二阶段（复活后）：超级玛卡巴卡：*扣除自身5点生命值(最低扣至0.01)，造成200%攻击力的光属性伤害，有30%几率再次发动技能\t\n    "
            "有1%的概率使出技能 玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡：*造成999点伤害"
        )
        self.skill1 = self.makabaka  # 设置技能指向
        self.skill2 = self.super_makabaka  # 设置技能指向

    def makabaka(self, opponent):
        self.damage = 0
        opponent.HP = opponent.HP
        self.HP += 2
        self.ATK += 5
        print(f"{self.name} 使用了 拍手 提高血量和攻击力")
        sleep(1)
        print(f"{self.name} 血量{self.HP} 攻击力{self.ATK}")
        sleep(1)

    def super_makabaka(self, opponent):
        super_maka_ATK = 0
        if self.count_of_reactivation == 1:
            print(f"{self.name} 使用了 玛卡巴卡")
            sleep(1)
            super_maka_ATK = self.ATK
        elif self.count_of_reactivation == 0:
            super_maka_ATK = 0
            self.HP -= 5
            if self.HP <= 0:
                self.HP = 0.01
            print(
                f"{self.name} 使用了 超级玛卡巴卡 扣除自身血量至{self.HP} 造成更高伤害"
            )
            sleep(1)
            while True:
                a_maka_ATK = self.ATK
                super_maka_ATK += a_maka_ATK
                if random.random() < 0.01:
                    print(
                        f"{self.name} 生气了（1%概率） 使用了 玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡玛卡巴卡，造成999点伤害"
                    )
                    sleep(2.5)
                    super_maka_ATK = 999
                    break
                elif random.random() < 0.7:
                    break
                print(f"{self.name} 获得祝福 再次使用超级玛卡巴卡")
                sleep(1)
        self.damage = super_maka_ATK - opponent.DEF


pikachu = Pikachu("皮卡丘", 80, 80, 35, 35, 5, "电", 0.3, "None")
pikachu2 = Pikachu("皮卡丘[电脑]", 80, 80, 35, 35, 5, "电", 0.3, "None")

bulbasaur = Bulbasaur("妙蛙种子", 100, 100, 35, 35, 10, "草", 0.1, "None")
bulbasaur2 = Bulbasaur("妙蛙种子[电脑]", 100, 100, 35, 35, 10, "草", 0.1, "None")

squirtle = Squirtle("杰尼龟", 80, 80, 25, 25, 25, "水", 0.2, "None")
squirtle2 = Squirtle("杰尼龟[电脑]", 80, 80, 25, 25, 25, "水", 0.2, "None")

charmander = Charmander("小火龙", 60, 60, 30, 30, 15, "火", 0.15, "None")
charmander2 = Charmander("小火龙[电脑]", 60, 60, 30, 30, 15, "火", 0.15, "None")

makabaka = Makabaka("玛卡巴卡", 20, 30, 20, 20, 10, "光", 0.2, "None")
makabaka2 = Makabaka("玛卡巴卡[电脑]", 20, 30, 20, 20, 10, "光", 0.2, "None")


# 属性被动技能调用函数
def UseWaterPassive(role, opponent):  # 水属性
    if role.property == "水":
        role.water_passive(opponent)


def UseGrassPassive(role):  # 草属性
    if role.property == "草":
        role.grass_passive()


def UseFirePassive(role, opponent):  # 火属性
    if role.property == "火":
        role.fire_passive(opponent)


def UseUserElectricalPassive(role, opponent):  # 电属性（玩家）
    if role.property == "电":
        role.user_electrical_passive(opponent)


def UseComputerElectricalPassive(role, opponent):  # 电属性（电脑）
    if role.property == "电":
        role.computer_electrical_passive(opponent)


def UseLightPassive(role):  # 光属性
    if role.property == "光":
        role.light_passive()


# 回合函数
def UserTerm(user_chosen_local, computer_chosen_local):
    global user_team, user_chosen_number, user_pokemon_dictionary_str_2, user_chosen
    global computer_team, computer_chosen_number, computer_chosen
    ##### 我方回合
    print(
        f"{user_chosen_local.name}的技能为：\n    1.{user_chosen_local.skill1_str}\n    2.{user_chosen_local.skill2_str}"
    )
    user_chosen_local.skill_dict = {
        1: user_chosen_local.skill1,
        2: user_chosen_local.skill2,
    }
    while True:
        try:
            user_skill_number = int(input("请输入您要使用的技能"))
            user_chosen_local.skill = user_chosen_local.skill_dict[user_skill_number]
            break
        except ValueError:
            print("输入无效 请重新输入")
        except KeyError:
            print("输入无效 请重新输入")
    ##### 判断是否闪避概率增加 （小火龙技能、玛卡巴卡被动）
    # 小火龙技能
    if (
        user_chosen_local.skill == charmander.flame_charge
        and user_chosen_local.flame_charge_status == "Preparation"
    ):
        print(f"{computer_chosen_local.name} 即将面对蓄能爆炎 闪避率增加20%")
        sleep(1)
        computer_chosen_local.dodge_probability += 0.2
        if computer_chosen_local.dodge_probability > 1:
            computer_chosen_local.dodge_probability = 1
    # 玛卡巴卡被动
    if computer_chosen_local == makabaka2 and computer_chosen_local.HP <= 5:
        computer_chosen_local.dodge_probability += 0.3
        print("玛卡巴卡[电脑]的血量低于5点 闪避率增加0.3")
        sleep(1)
    ##### 判断闪避 以便后续部分效果能用闪避判断是否生效
    computer_chosen_local.dodge_judgement = computer_chosen_local.dodge()
    ##### 闪避概率重置（小火龙技能、玛卡巴卡被动）
    # 小火龙闪避重置
    if (
        user_chosen_local.skill == charmander.flame_charge
        and user_chosen_local.flame_charge_status == "Preparation"
    ):
        computer_chosen_local.dodge_probability -= 0.2
    # 玛卡巴卡闪避重置
    if computer_chosen_local == makabaka2 and computer_chosen_local.HP <= 5:
        computer_chosen_local.dodge_probability -= 0.3
    ##### 使用技能
    user_chosen_local.skill(computer_chosen_local)
    if user_chosen_local.damage < 0:
        user_chosen_local.damage = 0
    if computer_chosen_local.dodge_judgement == True and user_chosen_local.damage > 0:
        print(f"{computer_chosen_local.name}闪避了{user_chosen_local.name}的攻击")
        sleep(1)
    elif (
        computer_chosen_local.dodge_judgement == True and user_chosen_local.damage == 0
    ):
        print()
    # 电脑 电属性被动判断——判断技能再次发动
    if computer_chosen_local.dodge_judgement == True and user_chosen_local.damage > 0:
        UseComputerElectricalPassive(computer_chosen_local, user_chosen_local)  # 电脑
        computer_chosen_local.dodge_judgement = False
    else:
        ####### 判断属性克制对伤害的影响
        user_chosen_local.restrain(computer_chosen_local)
        # 玛卡巴卡被动：受到技能伤害不超过10
        if computer_chosen_local == makabaka2 and user_chosen_local.damage > 10:
            user_chosen_local.damage = 10
        # 水属性被动判断——判断伤害减免
        UseWaterPassive(computer_chosen_local, user_chosen_local)
        # 判断杰尼龟技能
        computer_chosen_local.shield_judgement = ShieldJudgement(
            computer_chosen_local, user_chosen_local
        )
        ShieldJudgement(computer_chosen_local, user_chosen_local)
        ######### 输出伤害文字
        print(
            f"{user_chosen_local.name} 对 {computer_chosen_local.name} 造成了 {user_chosen_local.damage} 点伤害 "
        )
        ShieldReflect(computer_chosen_local, user_chosen_local)
        sleep(1)

        ####### 计算对手剩余血量
        computer_chosen_local.HP = computer_chosen_local.HP - user_chosen_local.damage
        if computer_chosen_local.HP <= 0:
            computer_chosen_local.HP = 0
        print(f"{user_chosen_local.name}剩余HP:{user_chosen_local.HP}")
        print(f"{computer_chosen_local.name}剩余HP:{computer_chosen_local.HP}")
        sleep(0.8)
        print()
        # 光属性被动判断——判断复活
        UseLightPassive(computer_chosen_local)
        # 火属性被动判断——判断伤害叠加
        UseFirePassive(user_chosen_local, computer_chosen_local)
    ##### 当有一方阵亡时其重新选择角色继续对战
    DeadJudgement(user_chosen_local, computer_chosen_local)


def ComputerTerm(user_chosen_local, computer_chosen_local):
    global user_team, user_chosen_number, user_pokemon_dictionary_str_2, user_chosen
    global computer_team, computer_chosen_number, computer_chosen
    ##### 敌方回合
    print(
        f"{computer_chosen_local.name}的技能为：\n    1.{computer_chosen_local.skill1_str}\n    2.{computer_chosen_local.skill2_str}"
    )
    computer_chosen_local.skill_dict = {
        1: computer_chosen_local.skill1,
        2: computer_chosen_local.skill2,
    }
    sleep(1)
    computer_skill_number = random.randint(1, 2)
    computer_chosen_local.skill = computer_chosen_local.skill_dict[
        computer_skill_number
    ]
    ##### 判断是否闪避概率增加 （小火龙技能、玛卡巴卡被动）
    # 小火龙技能
    if (
        computer_chosen_local.skill == charmander2.flame_charge
        and computer_chosen_local.flame_charge_status == "Preparation"
    ):
        user_chosen_local.dodge_probability += 0.2
        print(f"{user_chosen_local.name} 即将面对蓄能爆炎 闪避率增加20%")
        sleep(1)
        if user_chosen_local.dodge_probability > 1:
            user_chosen_local.dodge_probability = 1
    # 玛卡巴卡被动
    if user_chosen_local == makabaka and user_chosen_local.HP <= 5:
        user_chosen_local.dodge_probability += 0.3
        print("玛卡巴卡的血量低于5点 闪避率增加0.3")
        sleep(1)
    ##### 判断闪避 以便后续部分效果能用闪避判断是否生效
    user_chosen_local.dodge_judgement = user_chosen_local.dodge()
    ##### 闪避概率重置（小火龙技能、玛卡巴卡被动）
    # 小火龙闪避重置
    if (
        computer_chosen_local.skill == charmander2.flame_charge
        and computer_chosen_local.flame_charge_status == "Preparation"
    ):
        user_chosen_local.dodge_probability -= 0.2
    # 玛卡巴卡闪避重置
    if user_chosen_local == makabaka and user_chosen_local.HP <= 5:
        user_chosen_local.dodge_probability -= 0.3
    # 使用技能
    computer_chosen_local.skill(user_chosen_local)
    if computer_chosen_local.damage < 0:
        computer_chosen_local.damage = 0
    if user_chosen_local.dodge_judgement == True and computer_chosen_local.damage > 0:
        print(f"{user_chosen_local.name}闪避了{computer_chosen_local.name}的攻击")
        sleep(1)
    elif (
        user_chosen_local.dodge_judgement == True and computer_chosen_local.damage == 0
    ):
        print()
    # 玩家电属性被动判断——判断技能再次发动
    if user_chosen_local.dodge_judgement == True and computer_chosen_local.damage > 0:
        UseUserElectricalPassive(user_chosen_local, computer_chosen_local)  # 玩家
        user_chosen_local.dodge_judgement = False
    else:
        ######### 判断属性克制对伤害的影响
        computer_chosen_local.restrain(user_chosen_local)
        # 玛卡巴卡被动：受到技能伤害不超过10
        if user_chosen_local == makabaka and computer_chosen_local.damage > 10:
            computer_chosen_local.damage = 10
        # 水属性被动判断——判断伤害减免
        UseWaterPassive(user_chosen_local, computer_chosen_local)
        # 判断杰尼龟技能
        user_chosen_local.shield_judgement = ShieldJudgement(
            user_chosen_local, computer_chosen_local
        )
        ShieldJudgement(user_chosen_local, computer_chosen_local)
        print(
            f"{computer_chosen_local.name} 对 {user_chosen_local.name} 造成了 {computer_chosen_local.damage} 点伤害 "
        )
        ShieldReflect(user_chosen_local, computer_chosen_local)
        sleep(1)
        ######### 计算我方剩余血量
        user_chosen_local.HP = user_chosen_local.HP - computer_chosen_local.damage
        if user_chosen_local.HP <= 0:
            user_chosen_local.HP = 0
        print(f"{user_chosen_local.name}剩余HP:{user_chosen_local.HP}")
        print(f"{computer_chosen_local.name}剩余HP:{computer_chosen_local.HP}")

        sleep(0.8)
        print()
        # 光属性被动判断——判断复活
        UseLightPassive(user_chosen_local)
        # 火属性被动判断——判断伤害叠加
        UseFirePassive(computer_chosen_local, user_chosen_local)
    # 当有一方阵亡时其重新选择角色继续对战
    DeadJudgement(user_chosen_local, computer_chosen_local)


# 判断函数
def DeadJudgement(user_1_chosen, computer_1_chosen):
    # 死亡判断并重新选择角色
    # 死亡判断语句
    # DeadJudgement(user_chosen,computer_chosen)
    # if DeadJudgement(user_chosen,computer_chosen)==True:
    #     break
    global user_team, user_chosen_number, user_pokemon_dictionary_str_2, user_chosen
    global computer_team, computer_chosen_number, computer_chosen
    if user_1_chosen.HP <= 0:
        print(f"{user_1_chosen.name}阵亡，玩家派出下一个宝可梦")
        sleep(1.2)
        del user_team[user_chosen_number]
        del user_pokemon_dictionary_str_2[user_chosen_number]
        # 若全体阵亡，退出判断
        if len(user_team) == 0 or len(computer_team) == 0:
            print()
        else:
            # 重新输出玩家宝可梦队伍 并让玩家选择
            print(user_pokemon_dictionary_str_2)
            while True:
                try:
                    user_chosen_number = int(input("输入数字选择您的出战宝可梦:"))
                    user_chosen = user_team[user_chosen_number]
                    break
                except ValueError:
                    print("输入无效 请重新输入")
                except KeyError:
                    print("输入无效 请重新输入")
            print(f"您选择了 {user_pokemon_dictionary_str_2[user_chosen_number]}！")
            sleep(1.2)
            # 重新判断属性克制关系
            user_chosen.print_restrain(computer_chosen)
            return True, user_pokemon_dictionary_str_2, user_chosen_number, user_chosen
        return user_team
    elif computer_1_chosen.HP <= 0:
        print(f"{computer_1_chosen.name}阵亡，电脑派出下一个宝可梦")
        sleep(1.6)
        del computer_team[computer_chosen_number]
        # 若全体阵亡，退出判断
        if len(user_team) == 0 or len(computer_team) == 0:
            print()
        else:
            computer_chosen_number = random.choice(list(computer_team.keys()))
            computer_chosen = computer_team[computer_chosen_number]
            print("电脑选择中......")
            sleep(1.5)
            print(f"电脑选择了 {computer_chosen.name}！")
            sleep(1)
            # 重新判断属性克制关系
            user_chosen.print_restrain(computer_chosen)
            return True, computer_chosen_number, computer_chosen
        return computer_team


def StatusJudgement(role, opponent):
    # 状态判断及其效果（于回合初判断）
    # status_list=["Poisoned","Burning","Palsy","ParasiticOpponent"（妙蛙种子对自己施加这个状态）]
    for i in role.status_list:
        # “中毒”状态：每回合损失10%生命值，持续两回合
        if i == "Poisoned":
            role.poisoned_count -= 1
            poisoned_damage = 0.1 * role.HP
            role.HP -= poisoned_damage
            if role.HP <= 0:
                role.HP = 0
            print(
                f"{role.name} 处于中毒状态（还剩{role.poisoned_count}回合），损失{poisoned_damage}点生命值，还剩 {role.HP} 点生命值"
            )
            sleep(1)
            if role.poisoned_count == 0:
                role.status_list.remove("Poisoned")
        # “烧伤”状态：每回合受到10额外伤害， 持续2回合
        if i == "Burning":
            role.burning_count -= 1
            if role.burning_count >= 0:
                burning_damage = 10
                role.HP -= burning_damage
                if role.HP <= 0:
                    role.HP = 0
                print(
                    f"{role.name} 处于烧伤状态（还剩{role.burning_count}回合），损失{burning_damage}点生命值，还剩 {role.HP} 点生命值"
                )
                sleep(1)
            if role.burning_count <= 0:
                role.burning_count = 0
                role.status_list.remove("Burning")
        # 麻痹(使敌人跳过1回合)
        # 在对战模块开头判断：if role.skip_judge==True: 跳过/else 执行回合 然后都重置role.skip_judge=False
        if i == "Palsy":
            role.palsy_count -= 1
            role.skip_judge = True
            print(
                f"{role.name} 处于麻痹状态（还剩{role.palsy_count}回合），跳过一次行动"
            )
            sleep(1)
            if role.palsy_count == 0:
                role.status_list.remove("Palsy")
        # 寄生：每回合吸取对手10%的最大生命值并恢复自己
        if i == "ParasiticOpponent":  # 寄生对手
            role.parasitic_count -= 1
            if role.parasitic_count >= 0:
                parasitic_damage = 0.1 * opponent.HP
                role.HP += parasitic_damage
                opponent.HP -= parasitic_damage
                if role.HP > role.max_HP:
                    role.HP = role.max_HP
                if opponent.HP <= 0:
                    opponent.HP = 0
                print(
                    f"{role.name}寄生 {opponent.name}(还剩{role.parasitic_count}回合),吸取{opponent.name}{parasitic_damage}点生命值并回复自己\n"
                    f"{role.name}的生命值为{role.HP},{opponent.name}的生命值为{opponent.HP}"
                )
                sleep(2)
            if role.parasitic_count == 0:
                role.status_list.remove("ParasiticOpponent")


def ShieldJudgement(role, opponent):  # 杰尼龟护盾判断 #上面填电脑 下面填玩家
    for i in role.status_list:
        if i == "SquirtleShield":
            if opponent.damage > 0:
                print(f"{role.name}的护盾成功抵挡50%{opponent.name}的伤害 护盾解除")
                sleep(1)
                opponent.damage = opponent.damage / 2
                role.status_list.remove("SquirtleShield")
                return True


def ShieldReflect(role, opponent):
    if role.shield_judgement == True:
        if random.random() < 0.5:
            if random.random() < 0.4:
                shield_damage = opponent.damage * 2  # 20%概率弹反2倍受到的伤害
                print(
                    f"{role.name}成功弹反2倍受到的伤害（20%概率），对{opponent.name}造成{shield_damage}点伤害"
                )
            else:
                shield_damage = opponent.damage  # 50%概率弹反1倍受到的伤害
                print(
                    f"{role.name}成功弹反1倍受到的伤害（50%概率），对{opponent.name}造成{shield_damage}点伤害"
                )
            opponent.HP -= shield_damage
            if opponent.HP < 0:
                opponent.HP = 0
        role.shield_judgement = False


# 游戏运行模块
########################################################################################################################################
########################################################################################################################################
# 角色选择模组#############################################################################################################################
# 设置变量字典：便于赋予变量
Introduce()
user_pokemon_dictionary = {
    1: pikachu,
    2: bulbasaur,
    3: squirtle,
    4: charmander,
    5: makabaka,
}
computer_pokemon_dictionary = {
    1: pikachu2,
    2: bulbasaur2,
    3: squirtle2,
    4: charmander2,
    5: makabaka2,
}
# 玩家选择队伍
print("""1.皮卡丘(电属性) 2.妙蛙种子(草属性) 3.杰尼龟(水属性) 4.小火龙(火属性) 5，玛卡巴卡(光属性)\t
请选择3个宝可梦用于组成你的队伍：""")
# 设置字符串字典1：便于输出中文名称
user_pokemon_dictionary_str_1 = {
    1: "皮卡丘(电属性)",
    2: "妙蛙种子(草属性)",
    3: "杰尼龟(水属性)",
    4: "小火龙(火属性)",
    5: "玛卡巴卡(光属性)",
}
computer_pokemon_dictionary_str_1 = {
    1: "皮卡丘[电脑](电属性)",
    2: "妙蛙种子[电脑](草属性)",
    3: "杰尼龟[电脑](水属性)",
    4: "小火龙[电脑](火属性)",
    5: "玛卡巴卡[电脑](光属性)",
}
# 设置变量字典：便于赋予变量
while True:
    try:
        user_team_choice_a = int(input("输入数字选择您的宝可梦:"))
        user_team_choice_b = int(input("输入数字选择您的宝可梦:"))
        user_team_choice_c = int(input("输入数字选择您的宝可梦:"))
        # 确保变量互不相等且都在字典中
        if (
            user_team_choice_a != user_team_choice_b
            and user_team_choice_b != user_team_choice_c
            and user_team_choice_a != user_team_choice_c
        ):
            print(f"""您的队伍为:\t
            1.{user_pokemon_dictionary_str_1[user_team_choice_a]} 2.{user_pokemon_dictionary_str_1[user_team_choice_b]} 3.{user_pokemon_dictionary_str_1[user_team_choice_c]}\t""")
            break
        else:
            print("输入无效 请重新输入")
    except ValueError:
        print("输入无效 请重新输入")
    except KeyError:
        print("输入无效 请重新输入")
print()
# 电脑选择队伍
computer_team_choice_a = None
computer_team_choice_b = None
computer_team_choice_c = None
print("轮到电脑选择宝可梦")
print("电脑选择宝可梦中......")
sleep(2)
print("电脑选择完毕")
while (
    computer_team_choice_a == computer_team_choice_b
    or computer_team_choice_b == computer_team_choice_c
    or computer_team_choice_a == computer_team_choice_c
):
    computer_team_choice_a = random.randint(1, 5)
    computer_team_choice_b = random.randint(1, 5)
    computer_team_choice_c = random.randint(1, 5)
    if (
        computer_team_choice_a
        and computer_team_choice_b
        and computer_team_choice_c in computer_pokemon_dictionary_str_1
        and computer_team_choice_a != computer_team_choice_b
        and computer_team_choice_b != computer_team_choice_c
        and computer_team_choice_a != computer_team_choice_c
    ):
        print(f"""电脑的队伍为:\t
        1.{computer_pokemon_dictionary_str_1[computer_team_choice_a]} 2.{computer_pokemon_dictionary_str_1[computer_team_choice_b]} 3.{computer_pokemon_dictionary_str_1[computer_team_choice_c]}\t""")
    else:
        continue
print("克制关系：水——→草——→火——→电——→光——→水")
####################设置变量队伍
user_team = {
    1: user_pokemon_dictionary[user_team_choice_a],
    2: user_pokemon_dictionary[user_team_choice_b],
    3: user_pokemon_dictionary[user_team_choice_c],
}
computer_team = {
    1: computer_pokemon_dictionary[computer_team_choice_a],
    2: computer_pokemon_dictionary[computer_team_choice_b],
    3: computer_pokemon_dictionary[computer_team_choice_c],
}
#################################设置字符串字典2
user_pokemon_dictionary_str_2 = {
    1: user_pokemon_dictionary_str_1[user_team_choice_a],
    2: user_pokemon_dictionary_str_1[user_team_choice_b],
    3: user_pokemon_dictionary_str_1[user_team_choice_c],
}
computer_pokemon_dictionary_str_2 = {
    1: computer_pokemon_dictionary_str_1[computer_team_choice_a],
    2: computer_pokemon_dictionary_str_1[computer_team_choice_b],
    3: computer_pokemon_dictionary_str_1[computer_team_choice_c],
}
# 玩家选择宝可梦
while True:
    try:
        user_chosen_number = int(input("输入数字选择您的出战宝可梦:"))
        user_chosen = user_team[user_chosen_number]
        break
    except ValueError:
        print("输入无效 请重新输入")
    except KeyError:
        print("输入无效 请重新输入")
print(f"您选择了 {user_chosen.name}！")
# 电脑选择宝可梦
computer_chosen_number = random.choice(list(computer_team.keys()))
computer_chosen = computer_team[computer_chosen_number]
print("轮到电脑选择出战宝可梦......")
sleep(1)
print(f"电脑选择了 {computer_chosen.name}！")
######################################################################
######################################################################
# 对战模块###############################################################
while len(user_team) > 0 and len(computer_team) > 0:
    # 初次属性判断 后续属性判断包含再DeadJudgement函数内
    if len(user_team) == 3 and len(computer_team) == 3:
        user_chosen.print_restrain(computer_chosen)

    while True:
        # 我方
        print("轮到我方回合")
        # 草属性被动判断
        UseGrassPassive(user_chosen)
        # 光属性被动判断——判断复活
        UseLightPassive(user_chosen)
        # 状态判断及效果
        StatusJudgement(user_chosen, computer_chosen)
        # 判断死亡
        DeadJudgement(user_chosen, computer_chosen)
        # 若全体阵亡 则退出循环（外循环判断不成功也会停止）
        if len(user_team) == 0 or len(computer_team) == 0:
            break
        # 判断是否被跳过回合
        if user_chosen.skip_judge == True:
            user_chosen.skip_judge = False
            print(input('输入"回车"以继续'))
        else:
            ############# 我方回合
            UserTerm(user_chosen, computer_chosen)
            # 若全体阵亡 则退出循环
            if len(user_team) == 0 or len(computer_team) == 0:
                break

        # 敌方
        print("轮到敌方回合")
        print(input('输入"回车"以继续'))
        # 草属性被动判断
        UseGrassPassive(computer_chosen)
        # 状态判断及效果
        StatusJudgement(computer_chosen, user_chosen)
        # 光属性被动判断——判断复活
        UseLightPassive(computer_chosen)
        # 判断死亡
        DeadJudgement(user_chosen, computer_chosen)
        # 若全体阵亡 则退出循环
        if len(user_team) == 0 or len(computer_team) == 0:
            break
        # 判断是否被跳过回合
        if computer_chosen.skip_judge == True:
            computer_chosen.skip_judge = False
            print(input('输入"回车"以继续'))
        else:
            ############# 敌方回合
            ComputerTerm(user_chosen, computer_chosen)
            # 若全体阵亡 则退出循环
            if len(user_team) == 0 or len(computer_team) == 0:
                break
# 循环结束判断胜负
if len(user_team) == 0:
    print("您队伍的成员已全部阵亡，很遗憾您输掉了此次对战，下次努力！！！")
elif len(computer_team) == 0:
    print("电脑队伍的成员已全部阵亡，恭喜您获得了对战胜利，再接再厉！！！")

