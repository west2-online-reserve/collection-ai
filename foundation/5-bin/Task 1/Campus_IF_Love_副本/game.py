import random
import sys
from typing import Dict

from story import DIALOGUES, GIFT_EFFECTS


class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogue = DIALOGUES.get(self.name, [])
        if dialogue:
            i = 0
            if self.affinity < 20:
                i = random.randint(0, 0)
            elif self.affinity < 40:
                i = random.randint(1, 1)
            elif self.affinity < 60:
                i = random.randint(1, 2)
            elif self.affinity < 80:
                i = random.randint(2, 3)
            else:
                i = random.randint(1, 3)

            print(dialogue[i]["text"])
            print(f"A: {dialogue[i]['optionA']}")
            print(f"B: {dialogue[i]['optionB']}")

            while True:
                choice = input("请输入选项：")
                if choice == "A":
                    self.change_affinity(5)
                    break
                elif choice == "B":
                    self.change_affinity(-5)
                    break
                else:
                    print("输入不合法，请选择 A 或 B 。")

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        default = GIFT_EFFECTS.get(gift, {}).get("default", 0)
        effect = GIFT_EFFECTS.get(gift, {}).get(self.name, default)
        self.change_affinity(effect)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")
        if self.affinity <= -20:
            print(f"你被 {self.name} 讨厌了，故事线结束。")
            sys.exit(0)
        elif self.affinity < 0:
            print(f"{self.name} 对你有些不满，请多加注意。")

    def check_ending(self) -> bool:
        if self.affinity >= 90:
            print(f"\n你和 {self.name} 的故事进入了关键节点！")
            dialogue = DIALOGUES.get(self.name, [])
            if dialogue:
                i = 4
                print(dialogue[i]["text"])
                print(f"A: {dialogue[i]['optionA']}")
                print(f"B: {dialogue[i]['optionB']}")
                while True:
                    choice = input("请输入选项：")
                    if choice == "A":
                        self.affinity = 100
                        print(
                            f"你成功get到 {self.name} 的心意，{self.name} 也对你心生好感。"
                        )
                        print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
                        return True
                    elif choice == "B":
                        self.affinity = 80
                        print(
                            f"你没有get到 {self.name} 的心意，{self.name} 显得有些失落。"
                        )
                        print(f"当前好感度：{self.affinity}")
                        return False
                    else:
                        print("输入不合法，请选择 A 或 B 。")
        return False


class Game:
    def __init__(self) -> None:
        self.characters: Dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target: Character = self.characters["None"]

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self) -> bool:
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice = input(
            "1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择："
        )
        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线
        if choice == "1":
            print("\n你耐心地为小白讲解算法题的解法，她的眼睛亮了起来。")
            print("小白：『原来是这样！谢谢你！』")
            print("你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看着你离去的背影，眼中闪过一丝失落。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print(
            "你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来..."
        )
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input(
            "1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择："
        )
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 要求同上
        if choice == "1":
            print("\n你耐心地向姐姐解释你的代码实现方法，她的眼神中闪过一丝赞赏。")
            print("姐姐：『原来如此，你的思路很清晰呢。』")
            print("你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("姐姐看着你专注的背影，微微叹了口气。")
            return False

    def story_loop(self) -> None:
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作

            # 输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.current_target.talk()

            # 输入2----
            elif choice == "2":
                print("你可以送的礼物有：")
                for gift in GIFT_EFFECTS.keys():
                    print(f"- {gift}")
                while True:
                    gift_choice = input("请输入你要送的礼物：")
                    if gift_choice in GIFT_EFFECTS:
                        self.current_target.give_gift(gift_choice)
                        break
                    else:
                        print("无效的礼物选择，请重新选择。")

            # 输入3----
            elif choice == "3":
                print(
                    f"{self.current_target.name} 的当前好感度：{self.current_target.affinity}"
                )

            # 输入4----
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            if (
                self.current_target == self.characters["None"]
                or self.current_target.check_ending()
            ):
                break
