import sys

from character import Character, GIFT_EFFECTS


class Game:
    def __init__(self) -> None:
        self.characters: dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "咖啡店里的研究生姐姐"),
            "None": Character("None", "无")
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
        if not self.scene_senpai():
            print("\n你离开了社团活动室，决定去图书馆转一转……")  # 学姐场景
            if not self.scene_xiaobai():
                print("\n你离开了图书馆，独自走到校外的咖啡店休息……")  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥, 眼前三妹子都不要？？死现充别玩 galgame")

    def scene_senpai(self) -> bool:
        print("\n【场景一: 社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        if choice == "1":
            print("\n你随手挑起一只笔, 在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.story_loop()
            if self.current_target == self.characters["None"]:
                return False 
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print("\n来，我教你一个小技巧！")
            print("小白眼睛一亮，你进入【小白线】")
            self.current_target = self.characters["小白"]
            self.story_loop()
            if  self.current_target == self.characters["None"]:
                return False
            return True
        else:
            print("这种简单的东西，你都不会？")
            print("你没有继续帮助她，转头离开了图书馆。")
            return False
        
    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你缓缓低眉， 指尖划过键盘， 看似毫不在意地开口， 慢慢讲起你的实现方法。")
            print("她安静听着， 嘴角噙着淡淡笑意， 目光一直落在你的身上。")
            print("'很有想法', 她轻声说， '要不要坐下来聊一会儿？'")
            print("你进入【姐姐线】")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            if  self.current_target == self.characters["None"]:
                return False
            return True
        else:
            print("姐姐没有再和你说话，转身离开了。")
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

           
            if choice == "1":
                self.current_target.talk()

            elif choice == "2":
                gifts = list(GIFT_EFFECTS.keys())

                print("\n你可以送这些礼物：")
                for index, gift_name in enumerate(gifts, start=1):
                    print(f"{index}. {gift_name}")

                gift_choice = input("请选择礼物编号：")

                if gift_choice.isdigit():
                    gift_index = int(gift_choice) - 1

                    if 0 <= gift_index < len(gifts):
                         gift = gifts[gift_index]
                         self.current_target.give_gift(gift)
                    else:
                        print("没有这个礼物编号。")
                else:
                    print("请输入数字。")
                

            elif choice == "3":
                print(f"{self.current_target.name} 当前好感度为：{self.current_target.affinity}")
                

            elif choice == "4":
                print("你选择离开，游戏结束")
                sys.exit(0)

            else:
                print("无效输入，请重新选择")

            if self.current_target.affinity < 0:
                print(f"{self.current_target.name}对你印象太差了，攻略失败！")
                print("1.改过自新，重新开始")
                print("2.开启下一位攻略对象")

                while True:
                    fail_choice = input("请输入你选择的数字：").strip()
                    if fail_choice == "1":
                        self.current_target.affinity = 0
                        self.current_target.dialogue_index = 0
                        print("重新开始攻略")
                        break
                    if fail_choice == "2":
                        print(f"{self.current_target.name} 看你的眼神越来越复杂……")
                        print("你后知后觉地发现：刚才那几句话，好像每一句都踩雷了。")
                        print(f"666兄弟你闹麻了，你的低情商把{self.current_target.name}无语走了。")
                        self.current_target = self.characters["None"]
                        break
                    print("输入无效，请重新选择。")

                if self.current_target == self.characters["None"]:
                    break
            if self.current_target.check_ending():
                break





           
