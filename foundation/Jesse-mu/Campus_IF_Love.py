import sys

GOOD_OPTIONS: dict[str, list[str]] = {"学姐": ["A", "B", "A", "A", "A"], "小白": ["A", "A", "A", "A", "A"], "姐姐": ["A", "A", "A", "A", "A"]}
GOOD_BONUS = 10
BAD_BONUS = -10
END_NEED = 100


class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        # 修复1: 初始化 talk_count，否则第一次聊天就会 AttributeError
        self.talk_count = 0

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        lines: list[dict[str, str]] = DIALOGUES.get(self.name, [])
        if not lines:
            print(f"{self.name}没有和你想说的话。")
            return
        index: int = self.talk_count
        if index >= len(lines):
            print(f"{self.name}今天就先聊到这吧。")
            print("(话题已经聊完了，换个方式增加好感度吧。)")
            return
        self.talk_count += 1
        line = lines[index]

        # 修复6: 选项始终显示；好感度>=60 只是额外的一句话
        if self.affinity >= 60:
            print(f"{self.name}看你的眼神和最初不一样了")
        print(f"{self.name}:{line['text']}")
        print(f"      A.{line['optionA']}")
        print(f"      B.{line['optionB']}")

        while True:
            pick = input("你的回答(A/B):").strip().upper()
            if pick in ("A", "B"):
                break
            print("只能输入 A 或 B,请重新输入")

        # 修复3: 好/坏答案判定移到 while 循环结束(拿到合法输入之后)，保证 bonus 一定有值
        good: str = GOOD_OPTIONS.get(self.name, ["A"] * len(lines))[index]
        if pick == good:
            bonus: int = GOOD_BONUS
            print(f"{self.name}愣了一会，随即轻轻笑开，看你的眼神柔和了一些。")
        else:
            bonus = BAD_BONUS
            print(f"{self.name}的表情淡了下去，空气安静了几秒，气氛有些尴尬。")

        self.change_affinity(5)
        self.change_affinity(bonus)

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        effects = GIFT_EFFECTS.get(gift)
        if effects is None:
            print("没有这件礼物。")
            return
        value = effects.get(self.name, effects.get("default", 0))
        self.change_affinity(value)

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        # 修复: 使用常量 END_NEED，而不是写死 100
        if self.affinity >= END_NEED:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target = self.characters["None"]

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
                    print("\n啥，眼前三妹子都不要？？死现充别玩 galgame")

    def scene_senpai(self):
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

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了 1 时 输出了 print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线
        if choice == "1":
            print("\n你霸气的夺过她的笔，在她诧异的目光下，你寥寥几笔就点破了算法的关键所在。")
            print("小白看了半天，目光一震，转而用崇拜的眼神看向你。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("你熟视无睹，默默离开。")
            return False

    def scene_jiejie(self):
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        # 要求同上
        if choice == "1":
            print("\n你气定神闲，将实现方法娓娓道来")
            print("姐姐双手托腮，面露微笑，目不转睛盯着你。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你旁若无人，继续敲着你的代码。")
            return False

    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：")

            # 修复4: 用 if/elif 串联，避免处理完 1/2/3 后又落入 else 打印“无效输入”
            # 修复5: 选项3 是“查看好感度”，改为打印，不再误调 change_affinity
            if choice == "1":
                self.current_target.talk()

            elif choice == "2":
                # 修复2: give_gift 需要 gift 参数，先询问送什么
                gifts = list(GIFT_EFFECTS.keys())
                print(f"可选礼物：{'、'.join(gifts)}")
                gift = input("送什么礼物：").strip()
                self.current_target.give_gift(gift)

            elif choice == "3":
                print(f"{self.current_target.name} 当前好感度：{self.current_target.affinity}")

            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            if self.current_target == self.characters["None"] or self.current_target.check_ending():
                break


DIALOGUES = {
    "学姐": [
        {"text": "你是新来的吧？画画过吗？", "optionA": "其实我更擅长写代码，但我愿意试试画画。", "optionB": "画画？那种小孩子的东西我可没兴趣。"},
        {"text": "这幅画……构图还算不错，你是随便画的吗？", "optionA": "嗯，我是随手画的，没想到能得到你的认可。", "optionB": "不，我可是很认真的！因为你在看嘛。"},
        {"text": "加入社团后，不能三天打鱼两天晒网。", "optionA": "放心吧，我会认真对待。", "optionB": "社团只是玩玩而已吧？不要太严肃了。"},
        {"text": "我平时比较严格，你会不会觉得我很不好相处？", "optionA": "严格才好啊，我喜欢有目标的人。", "optionB": "嗯……确实有点难相处。"},
        {"text": "今天的社团活动结束了，你要不要和我一起收拾？", "optionA": "当然可以，我来帮你。", "optionB": "算了吧，我还有事要做。"}
    ],
    "小白": [
        {"text": "Python 里的 for 循环，我总是写不对……", "optionA": "来，我教你一个小技巧！", "optionB": "这种简单的东西，你都不会？"},
        {"text": "你看我借了好多书，会不会显得很傻？", "optionA": "不会啊，这说明你很爱学习，很可爱。", "optionB": "嗯……的确有点太贪心了。"},
        {"text": "写代码的时候，我总喜欢喝奶茶……你呢？", "optionA": "我也是！来，下次我请你一杯。", "optionB": "我只喝水，健康第一。"},
        {"text": "你会不会觉得我太依赖你了？", "optionA": "依赖我也没关系，我喜欢这种感觉。", "optionB": "嗯……是有点吧。"},
        {"text": "要不要一起留在图书馆自习？我想有人陪。", "optionA": "好啊，我正好也要复习。", "optionB": "算了，我还是回宿舍打游戏吧。"}
    ],
    "姐姐": [
        {"text": "你也喜欢在咖啡店看书吗？", "optionA": "是啊，这里很安静，很适合思考。", "optionB": "其实我只是随便找个地方坐。"},
        {"text": "研究生的生活，其实没有你想象的那么光鲜。", "optionA": "我能理解，压力一定很大吧。", "optionB": "那我还是不要考研了。"},
        {"text": "你觉得，什么样的人最值得依靠？", "optionA": "稳重冷静的人。", "optionB": "长得好看的人。"},
        {"text": "我常常一个人坐到很晚，你不会觉得孤单吗？", "optionA": "有时候孤单是种享受。", "optionB": "一个人太寂寞了，我受不了。"},
        {"text": "如果有一天，我遇到困难，你会帮我吗？", "optionA": "当然会，你不用一个人扛着。", "optionB": "我大概帮不了你吧。"}
    ]
}


GIFT_EFFECTS = {
    # 通用 / 默认 值
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"default": -10},  # 所有人 -10
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}


if __name__ == "__main__":
    game = Game()
    game.start()
