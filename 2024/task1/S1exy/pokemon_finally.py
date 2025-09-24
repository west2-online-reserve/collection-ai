# 使用的解释器为 python 3.12 版本


import random
import time

# 全局函数定义区
a1 = 0
a2 = 0
a3 = 0
a4 = 0

class Pokemon:
    def __init__(self, name, hp, attack, defense, element, dodge_rate):
        self.name = name             #名字
        self.max_hp = hp            #最大血量
        self.hp = hp                #血量
        self.attack = attack        #攻击力
        self.defense = defense      #防御
        self.element = element      #属性
        self.dodge_rate = dodge_rate#躲闪率
        self.status_effects = []    #当前状态
        self.move = []              #技能
        self.protime = 1
        self.control = []

    # 草元素回血
    def add_hp(self):
        if self.element == "草":
            self.hp = self.hp * 1.1
            self.hp = round(self.hp, 2)
            print(f"{self.name}触发了草元素被动回复了{ self.hp*0.1 }点血,现在的血量为{self.hp}")

    # 火元素增伤
    def add_attack(self):
        i = 0
        if i == 4:
            print("{self.name}的增伤已经达到极限40%了")
        else:
            if self.element == "火":
                self.attack = self.attack * 1.1
                self.attack = round(self.attack,2)
                print(f"{self.name}触发了火元素被动 攻击力增加了{self.attack / 1.1 * 0.1} 现在攻击力为{self.attack}")
                i = i + 1

    # 闪避值判断
    def judge(self,other):
        """闪避值判断："""
        b = random.randint(1 , 100)
        if b <= other.dodge_rate:
            return False
        if b > other.dodge_rate:
            return True

    # 元素判断进行伤害加减
    def yuansu(self,other):
        """元素对于攻击力的判断"""
        if self.element == "电":
            #增加攻击力
            if other.element == "水":
                self.protime *= 2
                print(f"你的 {self.element} 克制对方的 {other.element} 元素 你的伤害翻倍了！")
            if other.element == "草":
                self.protime /= 2
                print(f"你的 {self.element} 被对方的 {other.element} 元素克制 你的伤害减少了50%")

        if self.element == "草":
            #增加攻击力
            if other.element == "水":
                self.protime *= 2
                print(f"你的 {self.element} 克制对方的 {other.element} 元素 你的伤害翻倍了！")
            if other.element == "火":
                self.protime /= 2
                print(f"你的 {self.element} 被对方的 {other.element} 元素克制 你的伤害减少了50%")

        if self.element == "火":
            #增加攻击力
            if other.element == "草":
                self.protime *= 2
                print(f"你的 {self.element} 克制对方的 {other.element} 元素 你的伤害翻倍了！")
            if other.element == "水":
                self.protime /= 2
                print(f"你的 {self.element} 被对方的 {other.element} 元素克制 你的伤害减少了50%")

        if self.element == "水":
            #增加攻击力
            if other.element == "火":
                self.protime *= 2
                print(f"你的 {self.element} 克制对方的 {other.element} 元素 你的伤害翻倍了！")
            if other.element == "电":
                self.protime /= 2
                print(f"你的 {self.element} 被对方的 {other.element} 元素克制 你的伤害减少了50%")

    # 元素攻击进行回调
    def reyuansu(self,other):
        self.protime = 1
        other.protime = 1

    #状态判定
    def status(self,other,n):
        if "中毒状态" in other.status_effects:
            print(f"{other.name}目前处于中毒状态生命值每回合减少10%,本回合减少了{other.hp * 0.9}点")
            other.hp *= 0.9

        if "寄生状态" in other.status_effects:
            global a1
            if a1 == 0:
                a1 = n + 2
            if n < a1:
                print(f"{other.name}目前处于被寄生状态，损失5%生命值并为妙蛙种子回复等量生命值")
                print(f"{other.name}损失了{other.hp * 0.05}生命值，妙蛙种子回复{other.hp * 0.05}点生命值")
                self.hp = self.hp + (other.hp * 0.05)
                other.hp = other.hp * 0.95
                print(f"{self.name} {self.hp} {other.name}  {other.hp}")
            if n == a1:
                print(f"{other.name}目前处于被寄生状态，损失5%生命值并为妙蛙种子回复等量生命值")
                print(f"{other.name}损失了{(other.hp * 0.05)}生命值，妙蛙种子回复{other.hp * 0.05}点生命值")
                self.hp = self.hp + (other.hp * 0.05)
                other.hp = other.hp * 0.95
                other.status_effects.remove("寄生状态")

                # 三回合持续

        if "水流护盾" in self.status_effects:
            print("杰尼龟使用了水流保护了自己，该回合伤害减少50%")
            other.protime *= 0.5
            self.status_effects.remove("水流护盾")
            #一回合

        if "烧伤" in other.status_effects:
            global a2
            if a2 == 0:
                a2 = n + 1
                print(f"{other.name}带有火花效果，本回合受到的伤害增加10%")
                self.protime *= 1.1
            if n <= a2:
                print(f"{other.name}带有火花效果，本回合受到的伤害增加10%")
                self.protime *= 1.1
                other.status_effects.remove("烧伤")

            # 两回合持续
        if "蓄能爆炎" in self.status_effects:
            print("小火龙释放了蓄能爆炎！")
            if p(80):
                print(f"{other.name}被烧伤了")
                other.status_effects.append("烧伤")
            tem = other.dodge_rate
            other.dodge_rate += 20
            self.protime *= 2
            self.attack_move(other,1)
            other.dodge_rate -= 20





    # 谁的技能判定：
    def who(self,other,n):
        if self.name == "皮卡丘":
            if n == 0:
                if p(100):
                    print(f"{other.name}被十万伏特电麻了伤害增加20%")
                    self.protime += 0.2
                self.attack_move(other,n)
            if n == 1:
                self.attack_move(other, n)
                if p(10):
                    print(f"{self.name}触发了快速攻击多进行一次攻击！")
                    self.attack_move(other, n)
        if self.name == "妙蛙种子":
            if n == 0:
                self.attack_move(other, n)
                if p(15):
                    print(f"{other.name}被种子炸弹炸到了，陷入了中毒状态")
                    other.status_effects.append("中毒状态")
                    #记得更改从下个回合之后才开始每回合损失10%生命值的设定
            if n == 1:
                print("妙蛙种子向对手播种，每回合吸取对手10%的最大生命值并恢复自己, 效果持续3回合")
                other.status_effects.append("寄生状态")
        if self.name == "杰尼龟":
            if n == 0:
                self.attack_move(other,n)
            if n == 1:
                print(f"杰尼龟使用水流形成保护盾，减少下一回合受到的伤害50%")
                self.status_effects.append("水流护盾")

        if self.name == "小火龙":
            if n == 0:
                self.attack_move(other, n)
                if p(10):
                    print(f"{other.name}陷入了烧伤状态，每回合受到10额外伤害， 持续2回合")
                    other.status_effects.append("烧伤")
            if n == 1:
                self.status_effects.append("蓄能爆炎")
                print(f"开始蓄力，蓄能爆炎,下个回合才能进行释放")


    # 伤害大模块
    def attack_move(self, other, n):

        # 闪避判断
        if self.judge(self):
            # 元素克制判断
            self.yuansu(other)
            damage = self.attack * self.move[n]["time"] * self.protime - other.defense
            if damage < 0:
                damage = 0
                print("此次攻击造成了 0 点伤害 未能击穿敌军护甲！")
            else:
                # 水元素减伤
                if other.element == "水":
                    damage = damage * 0.7
                    print(f"{other.name}触发了水元素被动 受到的伤害减少了{damage / 0.7 * 0.3} 现在受到的伤害为{damage}")
                other.hp = other.hp - damage
                print(f"{self.name} 对 {other.name} 造成了 {damage} 点伤害，太痛了")

                print(f"{other.name} 剩余血量：{other.hp}")
                # 火元素增伤
                self.add_attack()
        else:
            print(f"{self.name} 对 {other.name} 的攻击被闪避了哦")
            print(f"{other.name} 剩余血量：{other.hp}")

            # 电元素闪电五连板
            if other.element == "电":
                print(f"{other.name}闪避成功 触发了电元素被动 可以对敌人再次发动一次技能")
                if other.control[0] == "human":
                    print(f"你的 {other.name} 拥有如下技能:")
                    print(f"1. {other.move[0]["name"]} \n2. {other.move[1]["name"]}")
                    print("选择一个技能进行攻击:")
                    tem = int(input())
                    # 玩家攻击
                    print(f"{other.name} 使用了 {other.move[tem - 1]["name"]} !")
                    other.attack_move(self, tem - 1)
                if other.control[0] == "pc":
                    print(f"{other.name} 使用了 {other.move[random.randint(0, 1)]["name"]} !")
                    other.attack_move(self, random.randint(0, 1))
        # 元素攻击倍率回调
        self.reyuansu(other)

    # 结束模块：
    def ending(self,other):
        if self.hp <= 0:
            print(f"你的宝可梦 {self.name} 被打败了 陷入了昏厥 ")
            return 1
        if other.hp <= 0:
            print(f"电脑的宝可梦 {other.name} 被打败了 陷入了昏厥 ")
            return 2


class ElectricPokemon(Pokemon):               # 皮卡丘（电）
    def __init__(self, name, hp, attack, defense):
        super().__init__(name, hp, attack, defense, ' 电 ', 30)
        self.name = "皮卡丘"  # 名字
        self.max_hp = 80  # 最大血量
        self.hp = 80  # 血量
        self.attack = 35  # 攻击力
        self.defense = 5  # 防御
        self.element = "电"  # 属性
        self.dodge_rate = 30  # 躲闪率
        self.status_effects = []    #当前状态
        self.move = [{"name":"十万伏特","time": 1.4,"pro":"麻痹效果"},{"name":"电光一闪","time": 1.0,"pro":"快速攻击"}]
        self.protime = 1


    # 技能 1
    def Thunderbolt(self,other):
        other.status_effects.append("麻痹")

class GrassPokemon(Pokemon):
    def __init__(self, name, hp, attack, defense):
        super().__init__(name, hp, attack, defense, ' 草 ', 10)
        self.name = "妙蛙种子"  # 名字
        self.max_hp = 80  # 最大血量
        self.hp = 100  # 血量
        self.attack = 35  # 攻击力
        self.defense = 10  # 防御
        self.element = "草"  # 属性
        self.dodge_rate = 10  # 躲闪率
        self.status_effects = []  # 当前状态
        self.move = [{"name": "种子炸弹", "time": 1.0, "pro": "中毒"},
                     {"name": "寄生种子", "time": 0.0, "pro": "寄生"}]
        self.protime = 1

class WaterPokemon(Pokemon):
    def __init__(self, name, hp, attack, defense):
        super().__init__(name, hp, attack, defense, ' 水' , 20)
        self.name = "杰尼龟"  # 名字
        self.max_hp = 80  # 最大血量
        self.hp = 80  # 血量
        self.attack = 25  # 攻击力
        self.defense = 20  # 防御
        self.element = "水"  # 属性
        self.dodge_rate = 20  # 躲闪率
        self.status_effects = []  # 当前状态
        self.move = [{"name": "水枪", "time": 1.4, "pro": "无"},
                     {"name": "护盾", "time": 0, "pro": "护盾"}]
        self.protime = 1

class FirePokemon(Pokemon):
    def __init__(self, name, hp, attack, defense):
        super().__init__(name, hp, attack, defense, ' 火 ', 10)
        self.name = "小火龙"  # 名字
        self.max_hp = 80  # 最大血量
        self.hp = 80  # 血量
        self.attack = 35  # 攻击力
        self.defense = 10  # 防御
        self.element = "火"  # 属性
        self.dodge_rate = 10  # 躲闪率
        self.status_effects = []  # 当前状态
        self.move = [{"name": "火花", "time": 1.0, "pro": "烧伤"},
                     {"name": "蓄能爆炎", "time": 1.0, "pro": "烧伤pro"}]
        self.protime = 1

# 定义概率函数：
def p(n):
    """闪避值判断："""
    b = random.randint(1, 100)
    if b <= n:
        return True
    if b > n:
        return False

# 选择宝可梦第一个模块
def choose():    #挑选英雄函数
    bag = []
    c = ["皮卡丘","妙蛙种子","杰尼龟","小火龙"]
    print("请选择3个宝可梦用于组成你的队伍：")
    print("1.皮卡丘(电属性) 2.妙蛙种子(草属性) 3.杰尼龟(水属性) 4.小火龙(火属性)")
    print("请选择你的宝可梦（选择3只）： 中间以空格作为间隔")
    a = input().split()
    tem = []
    print("你选择的宝可梦分别是：")
    if (a[0] == "1") or (a[1] == "1") or (a[2] == "1"):
        print("1.皮卡丘(电属性)")
        tem.append(0)
    if (a[0] == "2") or (a[1] == "2") or (a[2] == "2"):
        print("2.妙蛙种子(草属性)")
        tem.append(1)
    if (a[0] == "3") or (a[1] == "3") or (a[2] == "3"):
        print("3.杰尼龟(水属性)")
        tem.append(2)
    if (a[0] == "4") or (a[1] == "4") or (a[2] == "4"):
        print("4.小火龙(火属性)")
        tem.append(3)


    print("请选择你的宝可梦：")
    print(f"1.{c[tem[0]]} 2.{c[tem[1]]} 3.{c[tem[2]]}")
    print("输入数字选择你的宝可梦：",end="")
    b = int(input())
    print(f"你选择了 {c[tem[b - 1]]}")

    numbers = [i for i in range( 0 , 3 ) if i not in [tem[b - 1]]]
    random_number = random.choice(numbers)

    print(f"电脑选择了：{c[random_number]}")
    return [tem[b - 1],random_number]

#游戏开始啦
def start():
    a = [
        ElectricPokemon(name="皮卡丘", hp=80, attack=35, defense=15),
        GrassPokemon(name="妙蛙种子", hp=80, attack=25, defense=20),
        WaterPokemon(name="杰尼龟", hp=100, attack=35, defense=10),
        FirePokemon(name="小火龙", hp=80, attack=35, defense=5)]
    choose_end = (choose())
    b = a[choose_end[0]]
    c = a[choose_end[1]]

    a[choose_end[0]].control.append("human")
    a[choose_end[1]].control.append("pc")

    # 回合开始：
    for i in range(1 , 100):
        print("---------------------------")
        print(f"这是第 {i} 个回合")


        #回血模组
        a[choose_end[0]].add_hp()

        # 状态判断模块 （技能特效）
        print("正在进行状态判断：...")
        time.sleep(1)
        a[choose_end[0]].status(a[choose_end[1]],i)
        a[choose_end[1]].status(a[choose_end[0]], i)
        print("---------------------------")

        #战斗模组
        fight(choose_end,a,i)

        # 判断结束（回合完全结束）
        tem = a[choose_end[0]].ending(a[choose_end[1]])
        if tem == 1:
            break
        if tem == 2:
            break


def time_sleep():
    """# 等待阶段 使函数暂停2s"""
    time.sleep(2)

#战斗模块
def fight(choose_end,a,i):
    # 小火龙模块
    if a[choose_end[0]].name == "小火龙":
        if "蓄能爆炎" in a[choose_end[0]].status_effects:
            a[choose_end[0]].status_effects.remove("蓄能爆炎")
            # 等待
            time_sleep()
            print("")
            # 电脑攻击
            nmb = random.randint(0, 1)
            print(f"{a[choose_end[1]].name} 使用了 {a[choose_end[1]].move[nmb]["name"]} !")
            a[choose_end[1]].who(a[choose_end[0]], nmb)
        else:
            print(f"你的 {a[choose_end[0]].name} 拥有如下技能:")
            print(f"1. {a[choose_end[0]].move[0]["name"]} \n2. {a[choose_end[0]].move[1]["name"]}")
            print("选择一个技能进行攻击:")
            tem = int(input())
            # 玩家攻击
            print(f"{a[choose_end[0]].name} 使用了 {a[choose_end[0]].move[tem - 1]["name"]} !")
            tem = tem - 1
            a[choose_end[0]].who(a[choose_end[1]], tem)
            # 等待
            time_sleep()
            print("")
            # 电脑攻击
            nmb = random.randint(0, 1)
            print(f"{a[choose_end[1]].name} 使用了 {a[choose_end[1]].move[nmb]["name"]} !")
            a[choose_end[1]].who(a[choose_end[0]], nmb)
    else:
        print(f"你的 {a[choose_end[0]].name} 拥有如下技能:")
        print(f"1. {a[choose_end[0]].move[0]["name"]} \n2. {a[choose_end[0]].move[1]["name"]}")
        print("选择一个技能进行攻击:")
        tem = int(input())
        # 玩家攻击
        print(f"{a[choose_end[0]].name} 使用了 {a[choose_end[0]].move[tem - 1]["name"]} !")
        tem = tem-1
        a[choose_end[0]].who(a[choose_end[1]],tem)
        # 等待
        time_sleep()
        print("")

        # 电脑攻击
        nmb = random.randint(0, 1)
        print(f"{a[choose_end[1]].name} 使用了 {a[choose_end[1]].move[nmb]["name"]} !")
        a[choose_end[1]].who(a[choose_end[0]], nmb)


        if a[choose_end[1]].name == "小火龙":
            if "蓄能爆炎" in a[choose_end[1]].status_effects:
                a[choose_end[1]].status_effects.remove("蓄能爆炎")
            else:
                nmb = random.randint(0,1)
                print(f"{a[choose_end[1]].name} 使用了 {a[choose_end[1]].move[nmb]["name"]} !")
                a[choose_end[1]].who(a[choose_end[0]],nmb)


# 主函数 全部实现的阶段：
def main():
    start()


main()
