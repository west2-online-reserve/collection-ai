import sys

class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target = None

    def start(self):
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

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
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
        if choice == "1":
            print("\n你略微思考后，便提笔给出了思路，引得小白惊叹。")
            print("小白向你投来钦佩的目光。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("在小白失望的眼神中，你快步离去。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你缓缓抬头，条理清晰的开始解释，让姐姐听的入了神。")
            print("姐姐对你的解释很感兴趣。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你头也不抬，专注的敲着代码，把姐姐晾在了一旁。")
        return False

    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")
            print("5. 存档")
            print("6. 读档")

            choice = input("请输入选项：")

            # 输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == '1':
                self.current_target.talk(DIALOGUES[self.current_target.name])


            # 输入2----
            elif choice == '2':
                print("你可以选择：鲜花 / 编程笔记 / 奶茶 / 奇怪的石头 / 精致的钢笔 / 可爱玩偶 / 夜宵外卖")
                gift = input("请选择礼物：").strip()
                self.current_target.give_gift(gift, GIFT_EFFECTS)


            # 输入3----
            elif choice == "3":
                print(f"{self.current_target.name} 当前好感度：{self.current_target.affinity}")

            if choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            elif choice == "5":
                save_data = {
                    "current_target": self.current_target.name,
                    "characters": {
                        name: {
                            "affinity": char.affinity,
                            "talk_count": char.talk_count
                        } for name, char in self.characters.items()
                    }
                }
                self.save_manager.save_game(save_data)

            elif choice == "6":
                load_data = self.save_manager.load_game()
                if load_data:
                    self.current_target = self.characters[load_data["current_target"]]
                    for name, data in load_data["characters"].items():
                        self.characters[name].affinity = data["affinity"]
                        self.characters[name].talk_count = data["talk_count"]
                    print(f"\n读档成功！当前目标：{self.current_target.name}")
            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                break



