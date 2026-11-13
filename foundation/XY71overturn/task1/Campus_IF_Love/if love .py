import sys

from If_love_story import Character          

from If_love_manage import  GIFT_EFFECTS


class Game:
    def __init__(self) -> None :
        self.characters = {
            "Azazel": Character("Azazel", "社团里的艺术少女"),
            "Apollyon": Character("Apollyon", "课堂上的元气同学"),
            "???": Character("???", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target = self.characters["None"]

    def start(self) -> None :
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai() == True:  # 学姐场景
            if not self.scene_xiaobai() == True:  # 小白场景
                if not self.scene_jiejie() == True:  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩 galgame")

    def scene_senpai(self) -> bool:
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，Azazel正拿着画板注意到你。")
        print("Azazel：『这位新生？要不要来试试？』")

        choice = input("A. 主动表现兴趣，拿起一只笔作画\nB. 表示抱歉，没兴趣，转身离开\n请选择：").upper()
        if choice == "A":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("Azazel目光一震，眼神变得格外认真。\n你进入【Azazel线】！")
            self.current_target = self.characters["Azazel"]
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：Apollyon】")
        print("你走进图书馆，发现 Apollyon 正在奋笔疾书，却被一道算法题难住了。")
        print("Apollyon：『呜呜……这题到底该怎么写呀？』")

        choice = input("A. 主动帮她解题\nB. 敷衍几句，转身离开\n请选择：").upper()
        if choice == "A":
            print("\n你耐心地为她讲解算法思路，Apollyon 眼睛一亮，露出感激的笑容。")
            print("Apollyon：『原来是这样！谢谢你！』\n你进入【Apollyon线】！")
            self.current_target = self.characters["Apollyon"]
            self.story_loop()
            return True
        else:
            print("Apollyon 失望地看着你离开，低声嘀咕：『唉……』")
            return False
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        #注意 除了判断外 你可以同时输出角色的反应 
        #比如在上一位角色的判断中 选择了 1 时 输出了 print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        #写法可以借鉴学姐线

    def scene_jiejie(self) -> bool:
        print("\n【场景三：???】")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("???：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("A. 主动帮她解题\nB. 敷衍几句，转身离开\n请选择：").upper()

        if choice == "A":
            print("\n你娓娓道来，姐姐的眼神中闪过一丝赞赏，她微微点头，露出温柔的笑容。")
            print("???：『原来如此，你的思路很独特呢。』\n你进入【???线】！")
            self.current_target = self.characters["???"]
            self.story_loop()
            return True
        else:
            print("姐姐略显失望地看了你一眼，然后转身离开。")
            return False
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        #要求同上
        
    def story_loop(self) -> None :
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("A. 和她聊天")
            print("B. 送她礼物")
            print("C. 查看好感度")
            print("D. 离开（退出游戏）")

            choice = input("请输入选项：").upper()

            # TODO 完成输入不同选项时 进行的操作 
            gifts = list(GIFT_EFFECTS.keys())
            choices_gift = {
                  "A": gifts[0],
                  "B": gifts[1],
                  "C": gifts[2],
                  "D": gifts[3],
                  "E": gifts[4],
                  "F": gifts[5],
                  "G": gifts[6]
            }

            #输入 1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "A":
                Character.talk(self.current_target)




            #输入 2----
            elif choice == "B":
                print("你决定要送什么礼物？")
                print(f"A. {gifts[0]}")
                print(f"B. {gifts[1]}")
                print(f"C. {gifts[2]}")
                print(f"D. {gifts[3]}")
                print(f"E. {gifts[4]}")
                print(f"F. {gifts[5]}")
                print(f"G. {gifts[6]}")
                gift_choice = input("请选择礼物：")
                Character.give_gift(self.current_target, choices_gift[gift_choice])
                



            #输入 3----
            elif choice == "C":
                print(f"{self.current_target.name} 的当前好感度为：{self.current_target.affinity}")
                




            elif choice == "D":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")
                continue
            if self.current_target == self.characters["None"] or self.current_target.check_ending() == True:
                break
            







if __name__ == "__main__": 
    game = Game()
    game.start()