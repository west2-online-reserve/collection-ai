from __future__ import annotations
import re
import time
import random
from typing import Literal

MyPokemon: Pokemon = None
ComputerPokemon: Pokemon = None
turn: int = 1


class Pokemon(object):
    name: str = "未知"
    attribute: Literal["草", "火", "水", "电", "一般","未知"] = "未知"
    is_getdamage: bool = False
    is_madedamage: bool = False
    is_dodge: bool = False
    jianshangxishu = 1
    spacialstatus: Literal["麻痹", "中毒", "烧伤", "正常"] = "正常"
    turn = 1
    is_Parasitic = 0
    is_sleep = 0
    skills = []
    action = 0
    specialaction = 0
    attacklevel = 0  # 能力等级
    defencelevel = 0
    speedlevel = 0
    dodgelevel = 0
    attackdebuff = 1.0
    defencedebuff = 1.0
    speeddebuff = 1.0
    is_man: bool = True

    def __init__(
        self,
        hp: int,
        attack: int,
        defence: int,
        dodge: float,
        experience: int,
        speed: int,
    ):
        self.hp = hp  # 最大生命值
        self.currenthp = self.hp
        self.attack = attack  # 初始攻击力
        self.currentattack = self.attack
        self.defence = defence
        self.currentdefence = self.defence
        self.dodge = dodge
        self.currentdodge = self.dodge
        self.speed = speed
        self.currentspeed = self.speed

    def specialattribute(self, opponent: Pokemon):  # 特性
        pass

    @classmethod
    def Attack(cls, me: Pokemon, opponent: Pokemon, damage: float):
        if not judgedodge(opponent.currentdodge):
            opponent.is_getdamage = True  # 受击判定
            opponent.specialattribute(me)  # 调用特性函数
            if judgedamage(me, opponent, damage):  # 攻击判定
                me.is_madedamage = True
                me.specialattribute(opponent)  # 调用特性函数
            return True
        else:
            dodge(me, opponent)  # 成功闪避
            return False

    def Action(self, opponent: Pokemon):
        self.skills[self.action].effect(self, opponent)


class electricalPokemon(Pokemon):
    attribute = "电"

    def specialattribute(self, opponent: Pokemon):
        if self.is_dodge:  # 再次进行攻击
            self.is_dodge = False

            if self.is_man:  # 是玩家
                print("触发特性 弹反,请选择一个技能进行攻击:")
                for i in range(1, len(self.skills)):
                    print(f"{i}. {self.skills[i].name}")
                action = self.action  # 储存之前要使用的招式
                self.action = checkinput(4)
                self.Action(opponent)
                self.action = action  # 恢复之前要使用的招式
            else:  # 是电脑
                print("触发特性 弹反 ,电脑正在选择技能...")
                time.sleep(2)
                action = self.action
                self.action = random.randint(1, 4)
                self.Action(opponent)
                self.action = action

class generalPokemon(Pokemon):
    attribute = "一般"

    def specialattribute(self, opponent: Pokemon):
        if self.is_getdamage:
            if judgedodge(0.3):
                print(f"{self.name} 触发特性嗜睡症")
                opponent.is_sleep = 2  # 被攻击有 30% 的几率让对方睡觉两回合
                time.sleep(1)
            self.is_getdamage = False

class waterPokemon(Pokemon):
    attribute = "水"

    def specialattribute(self, opponent: Pokemon):
        if self.is_getdamage:
            if judgedodge(0.5):  # 有 50% 的几率减免 30% 的伤害
                print(f"{self.name} 触发特性激流,减免 30% 的伤害")
                self.jianshangxishu -= 0.3
            self.is_getdamage = False


class firePokemon(Pokemon):
    attribute = "火"

    def specialattribute(self, opponent: Pokemon):
        if self.is_madedamage:
            if self.attacklevel < 4:
                print(f"成功命中,{self.name}增加一级攻击力")
                self.attacklevel += 1
                self.is_madedamage = False
            else:
                print("攻击力已经升到最高级")
        self.currentattack = int(
            (self.attack + self.attacklevel * 0.1 * self.attack) * self.attackdebuff
        )  # 计算总攻击力


class plantPokemon(Pokemon):
    attribute = "草"

    def specialattribute(self, opponent: Pokemon):  # 吃草
        global turn
        if turn - self.turn == 1:
            self.currenthp = int(self.currenthp + self.hp * 0.1)
            if self.currenthp > self.hp:
                self.currenthp = self.hp
            print(f"{self.name} 触发特性食草,恢复10%HP!{self.currenthp}/{self.hp}")
            time.sleep(1)


class skill(object):
    name: str = "未知"
    attribution: str = "未知"
    priority: int = 0

    @staticmethod  # 静态方法
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 无法使用技能")


class Thunderbolt(skill):
    name = "十万伏特"
    attribution = "电"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了 十万伏特！")
        damage = 1.4 * me.currentattack
        if Pokemon.Attack(me, opponent, damage):  # 攻击到了
            if judgedodge(0.1):  # 判定麻痹
                if is_zhengchang(opponent):
                    opponent.spacialstatus = "麻痹"
                    print(f"{opponent.name} 被麻痹了")


class Quick_Attack(skill):
    name = "电光一闪"
    attribution = "一般"
    priority = 2

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了 电光一闪！")
        damage = float(me.currentattack)
        Pokemon.Attack(me, opponent, damage)
        if judgedodge(0.1) and opponent.currenthp > 0:  # 判定是否进行下一次攻击
            print(f"{me.name} 再次使用了 电光一闪！")
            Pokemon.Attack(me, opponent, damage)


class Seed_Bomb(skill):
    name = "种子爆弹"
    attribution = "草"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了 种子炸弹！")
        damage = float(me.currentattack)
        if Pokemon.Attack(me, opponent, damage):  # 如果打到了
            if judgedodge(0.15):  # 判定中毒
                if is_zhengchang(opponent):
                    opponent.spacialstatus = "中毒"
                    print(f"{opponent.name} 中毒了")


class Parasitic_Seeds(skill):
    name = "寄生种子"
    attribution = "草"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了 寄生种子")
        print(f"{opponent.name}被寄生了")
        opponent.is_Parasitic = 3


class Aqua_Jet(skill):
    name = "水枪"
    attribution = "水"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了水枪！")
        damage = 1.5 * me.currentattack
        Pokemon.Attack(me, opponent, damage)  # 调用伤害函数造成伤害


class Shield(skill):
    name = "保护"
    attribution = "一般"
    priority = 3

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了保护！")
        me.jianshangxishu -= 0.5


class Ember(skill):
    name = "火花"
    attribution = "火"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了火花！")
        damage = me.currentattack * 1
        if Pokemon.Attack(me, opponent, damage):  # 如果打到了
            if judgedodge(0.1):  # 判定烧伤
                if is_zhengchang(opponent):
                    opponent.spacialstatus = "烧伤"
                    print(f"{opponent.name} 烧伤了")


class Flame_Charge(skill):
    name = "蓄能爆炎"
    attribution = "火"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        if me.specialaction == 0:
            print(f"{me.name} 使用了蓄能爆炎,正在蓄力")
            me.specialaction = 1
        elif me.specialaction == 1:
            print(f"{me.name} 召唤出强大火焰")
            damage = me.currentattack * 3
            opponent.dodge += 0.2  # 敌方闪避率增加 20%
            if Pokemon.Attack(me, opponent, damage):  # 如果打到了
                if judgedodge(0.8):  # 判定烧伤
                    if is_zhengchang(opponent):
                        opponent.spacialstatus = "烧伤"
                        print(f"{opponent.name} 烧伤了")
            opponent.dodge -= 0.2  # 敌方闪避率复原
            me.specialaction = 0  # 回到未蓄力状态


class Electromagneticwave(skill):
    name = "电磁波"
    attribution = "电"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了电磁波！")
        if is_zhengchang(opponent):
            opponent.spacialstatus = "麻痹"
            print(f"{opponent.name} 被麻痹了")
        else:
            print("电磁波失效了")


class Highly_toxic(skill):
    name = "剧毒"
    attribution = "毒"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了剧毒！")
        if is_zhengchang(opponent):
            opponent.spacialstatus = "中毒"
            print(f"{opponent.name} 中毒了")
        else:
            print("剧毒失效了")


class Diminish(skill):
    name = "变小"
    attribution = "一般"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了变小！{me.name} 的闪避提高了！")
        if me.dodgelevel < 4:
            me.dodgelevel += 1
            Calculatedodge(me)  # 重新计算闪避
        else:
            print(f"{me.name} 已经无法再变小了")


class Hot_water(skill):
    name = "热水"
    attribution = "水"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了热水！")
        damage = me.currentattack * 1.2
        if Pokemon.Attack(me, opponent, damage):  # 如果打到了
            if judgedodge(0.15):  # 判定烧伤
                if is_zhengchang(opponent):
                    opponent.spacialstatus = "烧伤"
                    print(f"{opponent.name} 烧伤了")


class Impregnable_fortress(skill):
    name = "铁壁"
    attribution = "钢"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了铁壁！")
        if me.defencelevel < 6:
            me.defencelevel += 2
            Calculatedefence(me)  # 重新计算防御值
        else:
            print("已经无法再提升了")


class Skull_bash(skill):
    name = "火箭头槌"
    attribution = "一般"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        if me.specialaction == 0:
            print(f"{me.name} 将头缩进龟壳中")
            me.specialaction = 1
            me.defencelevel += 2  # 提高防御力
            Calculatedefence(me)
        elif me.specialaction == 1:
            print(f"{me.name} 使用了火箭头槌!")
            damage = me.currentattack * 1.7
            Pokemon.Attack(me, opponent, damage)
            me.specialaction = 0  # 回到未蓄力状态
            me.defencelevel -= 2  # 减少防御力
            Calculatedefence(me)


class Flare_blitz(skill):
    name = "闪焰冲锋"
    attribution = "火"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了闪焰冲锋！")
        damage = me.currentattack * 2
        hp = opponent.currenthp
        Pokemon.Attack(me, opponent, damage)
        if opponent.currenthp < hp:
            me.currenthp = int(me.currenthp - (hp - opponent.currenthp) / 3)
            if me.currenthp < 0:
                me.currenthp = 0
            print(f"{me.name} 受到了反作用力的伤害,剩余HP:{me.currenthp}/{me.hp}")


class Spore(skill):
    name = "蘑菇孢子"
    attribution = "草"
    priority = 1

    @staticmethod
    def effect(me: Pokemon, opponent: Pokemon):
        print(f"{me.name} 使用了蘑菇孢子！")
        if opponent.is_sleep == 0 and opponent.attribute != "草":
            opponent.is_sleep = 5
            print(f"{opponent.name} 睡着了！")
        else:
            print("蘑菇孢子失效了！")


class PikaChu(electricalPokemon):
    name = "皮卡丘"
    skills = [skill(), Thunderbolt(), Quick_Attack(), Shield(), Electromagneticwave()]


class SeedBomb(plantPokemon):
    name = "妙蛙种子"
    skills = [skill(), Seed_Bomb(), Parasitic_Seeds(), Highly_toxic(), Spore()]


class Squirtle(waterPokemon):
    name = "杰尼龟"
    skills = [
        skill(),
        Impregnable_fortress(),
        Skull_bash(),
        Highly_toxic(),
        Hot_water(),
    ]


class Charmander(firePokemon):
    name = "小火龙"
    skills = [skill(), Ember(), Flame_Charge(), Shield(), Flare_blitz()]


class Jigglypuff(generalPokemon):
    name = "胖丁"
    skills = [skill(), Diminish(), Electromagneticwave(), Highly_toxic(),Quick_Attack()]


def is_zhengchang(pokemon: Pokemon):
    if pokemon.spacialstatus == "正常":
        return True
    else:
        return False


def dodge(me: Pokemon, opponent: Pokemon):
    print(f"{opponent.name} 躲避了这次攻击")
    opponent.is_dodge = True  # 成功闪避
    opponent.specialattribute(me)


def judgedodge(dodge: float):  # 是否判定成功
    if random.random() < dodge:
        return True
    else:
        return False


def judgedamage(me: Pokemon, opponent: Pokemon, damage: float):
    kezhi = is_counter(me.skills[me.action].attribution, opponent.attribute)
    if kezhi == "克制":
        damage = int(damage * 2 * opponent.jianshangxishu)
    elif kezhi == "被克制":
        damage = int(damage / 2 * opponent.jianshangxishu)
    elif kezhi == "没有克制关系":
        damage = int(damage * opponent.jianshangxishu)
    if me.skills[me.action].attribution == me.attribute:
        damage = int(damage * 1.2)  # 同属性加成
    return is_defence(opponent, damage, kezhi)


def is_counter(
    myattribute: Literal["草", "火", "水", "电", "未知"],
    opponentattribute: Literal["草", "火", "水", "电", "未知"],
) -> Literal["克制", "被克制", "没有克制关系"]:  # 判断属性克制
    to = myattribute + opponentattribute
    res = re.match("(草水|火草|火钢|水火|电水)", to)  # 克制
    if res:
        return "克制"
    res = re.match("(草火|火水|水电|水草|电草|毒草)", to)  # 被克制
    if res:
        return "被克制"
    return "没有克制关系"


def is_defence(
    opponent: Pokemon, damage: int, counter: Literal["克制", "被克制", "没有克制关系"]
):  # 判断防御和攻击的关系
    if opponent.currentdefence >= damage:
        print(
            f"{opponent.name}防住了这次攻击,剩余HP:{opponent.currenthp}/{opponent.hp}"
        )
        return False
    else:

        opponent.currenthp = opponent.currenthp + opponent.currentdefence - damage
        if opponent.currenthp < 0:
            opponent.currenthp = 0
        if counter == "克制":
            print(
                f"{opponent.name} 受到了 {damage-opponent.currentdefence} 点伤害！效果拔群！剩余HP:{opponent.currenthp}/{opponent.hp}"
            )
        elif counter == "被克制":
            print(
                f"{opponent.name} 受到了 {damage-opponent.currentdefence} 点伤害！效果不好！剩余HP:{opponent.currenthp}/{opponent.hp}"
            )
        elif counter == "没有克制关系":
            print(
                f"{opponent.name} 受到了 {damage-opponent.currentdefence} 点伤害！剩余HP:{opponent.currenthp}/{opponent.hp}"
            )
        return True


def choose(player: str, pokemon: Pokemon, num: int) -> Pokemon:
    if num == 1:
        pokemon = PikaChu(80, 35, 5, 0.3, 0, 90)
    elif num == 2:
        pokemon = SeedBomb(100, 35, 10, 0.1, 0, 45)
    elif num == 3:
        pokemon = Squirtle(80, 25, 20, 0.2, 0, 43)
    elif num == 4:
        pokemon = Charmander(80, 35, 15, 0.1, 0, 65)
    elif num == 5:
        pokemon = Jigglypuff(115, 20, 10, 0.2, 0, 20)
    print(f"{player}选择了 {pokemon.name}！")
    return pokemon


def checkinput(j):
    while True:
        x = input()
        res = re.match(f"[1-{j}]", x)  # 匹配1-j
        if res:
            return int(res.group())  # 字符串转数字
        else:
            print("非法输入，请重新输入")


def chooseyourpokemon():

    print("请选择 1 个宝可梦作为你的伙伴")
    print(
        "1. 皮卡丘（电属性） 2. 妙蛙种子（草属性） 3. 杰尼龟（水属性） 4. 小火龙（火属性）5. 胖丁（一般属性）"
    )
    print("输入数字选择你的宝可梦：", end="")
    global MyPokemon
    global ComputerPokemon
    num = checkinput(5)
    MyPokemon = choose("你", MyPokemon, num)
    ComputerPokemon = choose("电脑", ComputerPokemon, random.randint(1, 4))
    ComputerPokemon.is_man = False
    time.sleep(2)


def Calculatedebuff(pokemon: Pokemon):  # 计算负面状态
    if pokemon.spacialstatus == "麻痹":
        pokemon.speeddebuff = 0.5  # 速度减半

    elif pokemon.spacialstatus == "中毒":
        pokemon.currenthp = int(pokemon.currenthp - pokemon.hp * 0.125 - pokemon.turn * 5)  # 每回合损失最大HP的12.5%加上一个递增的值
        if pokemon.currenthp <= 0:
            pokemon.currenthp = 0
        print(
            f"{pokemon.name} 受到中毒的影响,减少{int(pokemon.hp*0.125+pokemon.turn*3)}点HP,当前HP:{pokemon.currenthp}/{pokemon.hp}"
        )
        time.sleep(1)

    elif pokemon.spacialstatus == "烧伤":
        pokemon.currenthp = int(pokemon.currenthp - pokemon.hp * 0.1)
        if pokemon.currenthp <= 0:
            pokemon.currenthp = 0
        print(
            f"{pokemon.name} 受到烧伤的影响,减少{int(pokemon.hp*0.1)}点HP,当前HP:{pokemon.currenthp}/{pokemon.hp}"
        )
        pokemon.attackdebuff = 0.5  # 攻击减半
        time.sleep(1)

    elif pokemon.spacialstatus == "正常":
        pokemon.attackdebuff = 1
        pokemon.defencedebuff = 1
        pokemon.speeddebuff = 1


def judgeParasitic_Seeds(me: Pokemon, opponent: Pokemon):
    if opponent.is_Parasitic:
        gethp = int(opponent.hp * 0.1)
        opponent.currenthp = opponent.currenthp - gethp
        if opponent.currenthp < 0:
            opponent.currenthp = 0
        print(
            f"{opponent.name} 受到种子的影响,减少{gethp}点HP,剩余HP:{opponent.currenthp}/{opponent.hp}"
        )
        me.currenthp += gethp
        if me.currenthp > me.hp:
            me.currenthp = me.hp
        time.sleep(1)
        print(f"{me.name} 回复{gethp}点HP,剩余HP:{me.currenthp}/{me.hp}")
        opponent.is_Parasitic -= 1
        time.sleep(1)


def is_Coma(pokemon: Pokemon) -> bool:
    if pokemon.currenthp <= 0:
        return True
    else:
        return False


def Calculate(me: Pokemon, opponent: Pokemon):
    host, guest = jugdespeed(me, opponent)
    judgeParasitic_Seeds(host, guest)
    if is_Coma(guest):
        return True, host
    judgeParasitic_Seeds(guest, host)
    if is_Coma(host):
        return True, guest
    Calculatedebuff(host)
    if is_Coma(host):
        return True, guest
    Calculateattack(host)
    Calculatedefence(host)
    Calculatespeed(host)
    Calculatedodge(host)
    Calculatejianshang(host)
    init(host)

    Calculatedebuff(guest)
    if is_Coma(guest):
        return True, host
    Calculateattack(guest)
    Calculatedefence(guest)
    Calculatespeed(guest)
    Calculatedodge(guest)
    Calculatejianshang(guest)
    init(guest)
    return False, None


def Calculateattack(pokemon: Pokemon):  # 计算能力值
    pokemon.currentattack = int(
        (pokemon.attack + pokemon.attacklevel * 0.1 * pokemon.attack)
        * pokemon.attackdebuff
    )


def Calculatedefence(pokemon: Pokemon):
    pokemon.currentdefence = int(
        (pokemon.defence + pokemon.defencelevel * 0.1 * pokemon.defence)
        * pokemon.defencedebuff
    )


def Calculatespeed(pokemon: Pokemon):
    pokemon.currentspeed = int(
        (pokemon.speed + pokemon.speedlevel * 0.1 * pokemon.speed) * pokemon.speeddebuff
    )


def Calculatedodge(pokemon: Pokemon):
    pokemon.currentdodge = pokemon.dodge + pokemon.dodgelevel * 0.1


def Calculatejianshang(pokemon: Pokemon):
    pokemon.jianshangxishu = 1


def init(host: Pokemon):
    host.is_dodge = False
    host.is_getdamage = False
    host.is_madedamage = False


def gameprepare(me: Pokemon, opponent: Pokemon):
    print(
        f"你的 {me.name} 的HP:{me.currenthp}/{me.hp},攻击力:{me.currentattack}/{me.attack},防御力:{me.currentdefence}/{me.defence}"
    )
    print(
        f"对手的 {opponent.name} 的HP:{opponent.currenthp}/{opponent.hp},攻击力:{opponent.currentattack}/{opponent.attack},防御力:{opponent.currentdefence}/{opponent.defence}"
    )
    print(f"你的 {MyPokemon.name} 的技能：")
    for i in range(1, len(me.skills)):
        print(f"{i}. {me.skills[i].name}")


def judgesleep_and_action(pokemon: Pokemon, opponent: Pokemon):
    if pokemon.is_sleep > 0 and judgedodge(0.65):  # 睡觉中有 65% 的几率继续睡觉
        pokemon.is_sleep -= 1
        print(f"{pokemon.name} 睡着了，无法行动！")
    else:
        if pokemon.is_sleep > 0:
            print(f"{pokemon.name} 醒了！")
            pokemon.is_sleep = 0
            time.sleep(1)
        if pokemon.spacialstatus == "麻痹" and judgedodge(0.25):  # 麻痹状态
            # 麻痹时有 25% 的几率无法行动
            print(f"{pokemon.name} 麻痹了，无法行动！")
            time.sleep(1)
        else:
            pokemon.Action(opponent)


def chooseaction(pokemon: Pokemon):
    if pokemon.specialaction == 0 and pokemon.is_man:  # 未在蓄力
        print("请选择一个技能：")
        pokemon.action = checkinput(4)
    elif pokemon.specialaction == 0 and pokemon.is_man == False:
        pokemon.action = random.randint(1, 4)
    else:
        print(f"{pokemon.name} 正在蓄力，无法使用其他技能")


def battal(pokemon1: Pokemon, pokemon2: Pokemon):
    if (
        pokemon1.skills[pokemon1.action].priority
        > pokemon2.skills[pokemon2.action].priority
    ):  # 比较技能优先级
        host = pokemon1
        guest = pokemon2
    elif (
        pokemon1.skills[pokemon1.action].priority
        < pokemon2.skills[pokemon2.action].priority
    ):
        host = pokemon2
        guest = pokemon1
    else:
        host, guest = jugdespeed(pokemon1, pokemon2)
    judgesleep_and_action(host, guest)  # 先手行动
    if is_Coma(guest):
        return True, host
    if is_Coma(host):
        return True, guest
    time.sleep(1)
    judgesleep_and_action(guest, host)  # 后手行动
    if is_Coma(host):
        return True, guest
    if is_Coma(guest):
        return True, host
    time.sleep(1)
    return False, None


def jugdespeed(pokemon1: Pokemon, pokemon2: Pokemon):
    if pokemon1.currentspeed >= pokemon2.currentspeed:  # 根据速度判定谁先手
        return pokemon1, pokemon2
    else:
        return pokemon2, pokemon1


def startgame():
    global MyPokemon
    global ComputerPokemon
    global turn
    while MyPokemon.currenthp > 0 and ComputerPokemon.currenthp > 0:

        print(f"----------第{turn}回合----------")
        gameprepare(MyPokemon, ComputerPokemon)
        chooseaction(MyPokemon)
        chooseaction(ComputerPokemon)
        j, winer = battal(MyPokemon, ComputerPokemon)
        if j:
            return winer
        j, winer = Calculate(MyPokemon, ComputerPokemon)  # 结算状态
        if j:
            return winer
        host, guest = jugdespeed(MyPokemon, ComputerPokemon)
        turn += 1
        host.specialattribute(guest)  # 调用双方的特性
        guest.specialattribute(host)
        host.turn = turn
        guest.turn = turn
        time.sleep(2)
