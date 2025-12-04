import sys, random, time

from typing import Dict, Optional

from story import DIALOGUES, GIFT_EFFECTS, ENDING

def print_by_words(text: str, delay: float = 0.05) -> None:
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()
    # reference from GPT-5
    # 逐字输出文本，参数delay控制每个字符之间打印的延迟


class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self) -> None:
        print_by_words(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        # 根据角色和对话池随机选择一条对话，并根据选项调整好感度
        dialog_pool = DIALOGUES[self.name]
        dialog_selected = random.choice(dialog_pool)
        print_by_words(self.name + ':' + dialog_selected['text'])
        print('选项A:' + dialog_selected['optionA'])
        print('选项B:' + dialog_selected['optionB'])
        option = input('请输入选项：').strip().upper()
        if option == 'B':
            if self.affinity >= 50:
                self.change_affinity(5)
            elif self.affinity >= 25:
                self.change_affinity(3)
            elif self.affinity >= 10:
                self.change_affinity(-3)
            else:
                self.change_affinity(0)

        else:
            if self.affinity >= 50:
                self.change_affinity(10)
            elif self.affinity >= 25:
                self.change_affinity(7)
            else:
                self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        # 根据GIFT_EFFECTS字典（礼物效果）调整好感度
        effect = GIFT_EFFECTS.get(gift)
        value = 0
        if effect is None:
            print('没有这个礼物')
        else:
            value = effect.get(self.name, effect.get('default', 0))
        self.change_affinity(value)

    def change_affinity(self, value: int) -> None:
        # 修改好感度并打印变化，并保证好感度不低于0
        self.affinity += value
        if self.affinity < 0:
            self.affinity = 0
        sign = '+' if value > 0 else ''
        print(f"{self.name} 的好感度变化 {sign}{value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print_by_words(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


class Game:
    def __init__(self) -> None:
        self.characters: Dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target: Optional[Character] = None

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print_by_words("你是一名刚刚踏入大学校园的新生。")
        print_by_words("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print_by_words("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print_by_words("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print_by_words("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print_by_words("\n啥，眼前三妹子都不要？？死现充别玩galgame", 0.01)

    def scene_senpai(self) -> bool:
        print_by_words("\n【场景一：社团学姐】")
        print_by_words("你路过社团活动室，学姐正拿着画板注意到你。")
        print_by_words("学姐：『这位新生？要不要来试试？』")

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        # 学姐的决定点场景，返回是否进入学姐线
        if choice == "1":
            print_by_words("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print_by_words("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.story_loop()
            return True
        else:
            print_by_words("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print_by_words("\n【场景二：小白】")
        print_by_words("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print_by_words("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线
        # 小白的决定点场景，返回是否进入小白线
        if choice == '1':
            print_by_words('\n你轻轻走到她身边，指着书上的代码耐心解释。')
            print_by_words('小白眼睛一亮：「原来是这样！谢谢你！」\n她的脸颊微微泛红，似乎对你产生了好感。你进入【小白线】！')
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print_by_words('你随口说了几句「多练练就会了」，便转身离开。\n小白愣了一下，低下头继续盯着书本，眼神黯淡。\n你错过了与她的缘分……')
            return False

    def scene_jiejie(self) -> bool:
        print_by_words("\n【场景三：姐姐")
        print_by_words("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print_by_words("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 要求同上
        # 姐姐的决定点场景，返回是否进入姐姐线
        if choice == '1':
            print_by_words('\n你抬起头，语气平和地解释了自己的代码逻辑。')
            print_by_words('姐姐认真倾听，眼神中闪过一丝欣赏：「原来如此，你比我想象的还要成熟呢。」\n她的语气温柔，却带着几分探究的意味。你进入【姐姐线】！')
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print_by_words('你假装没有听见，继续盯着屏幕。\n姐姐愣了一下，轻轻摇头，转身回到座位。\n咖啡店的氛围依旧安静，但你错过了最后的机会……')
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
            if choice == '1':
                self.current_target.talk()


            # 输入2----
            elif choice == '2':
                print('可选礼物：鲜花 / 编程笔记 / 奶茶 / 奇怪的石头 / 精致的钢笔 / 可爱玩偶 / 夜宵外卖')
                gift = input('请选择你要送的礼物（打字输入）：').strip()
                self.current_target.give_gift(gift)


            # 输入3----
            elif choice == '3':
                print(f'{self.current_target.name}当前好感度：{self.current_target.affinity}')



            elif choice == "4":
                print_by_words("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                print_by_words(ENDING.get(self.current_target.name, '出bug了'))
                break