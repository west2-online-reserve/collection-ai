import sys


class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.talk_count = 0

    def talk(self):
        """从全局 DIALOGUES 轮询对话，聊天有基础好感 +5，选项 A/B 额外影响。"""
        print(f"你正在和{self.name}对话...")

        dialogs = DIALOGUES.get(self.name, [])
        if not dialogs:
            print("她似乎没有想跟你说话。")
            return

        idx = self.talk_count % len(dialogs)
        dlg = dialogs[idx]
        print(dlg.get("text", ""))
        print("A.", dlg.get("optionA", ""))
        print("B.", dlg.get("optionB", ""))

        choice = input("请选择 (A/B)：").strip().lower()

        # 基础聊天好感
        delta = 5
        if choice == 'a':
            delta += 10
            print("她听了之后露出微笑，似乎很开心。")
        elif choice == 'b':
            delta += -5
            print("她听后表情有些淡，气氛略微僵了一下。")
        else:
            print("无效选择，按客套回答处理。")

        self.change_affinity(delta)
        self.talk_count += 1

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")

        effect_entry = GIFT_EFFECTS.get(gift)
        value = 0
        if effect_entry is None:
            print("她看着你手里的东西似乎有些困惑，不知道这是什么礼物。")
            value = 0
        else:
            # 如果对该角色有专门的设定则使用，否则使用 default（若存在）或 0
            if self.name in effect_entry:
                value = effect_entry[self.name]
            else:
                value = effect_entry.get('default', 0)

        # 可加入小段交互反馈
        if value > 0:
            print(f"她很喜欢这份礼物，好感提升 {value} 点。")
        elif value < 0:
            print(f"这份礼物让她有些不悦，好感下降 {-value} 点。")
        else:
            print("这份礼物没有太大影响。")

        self.change_affinity(value)

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target = None

    def start(self):
        # 开场文案（逐字显示在命令行上可以通过逐字符打印来模拟，但这里简化为普通打印）
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

    def scene_senpai(self):
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        if choice.strip() == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.story_loop()
            return True
        else:
            print("在众人注视下你选择离开。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice.strip() == "1":
            print("\n你坐下来耐心地给她讲解思路，终于把题目讲通了。小白激动地对你露出感激的笑容。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看了看你，有些失望地继续低头做题，你转身离开。")
            return False

    def scene_jiejie(self):
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice.strip() == "1":
            print("\n你温和地向她解释你的实现，她认真倾听，时不时点头。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("她看了看你，又低下头喝了口咖啡，你继续敲代码。")
            return False

    def story_loop(self):
        """角色线主循环：聊天/送礼/查看好感/离开"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：").strip()

            if choice == "1":
                if self.current_target is None:
                    print("当前没有角色可以互动。")
                else:
                    self.current_target.talk()

            elif choice == "2":
                if self.current_target is None:
                    print("当前没有角色可以送礼。")
                else:
                    gifts = list(GIFT_EFFECTS.keys())
                    print("可选择的礼物：")
                    for i, g in enumerate(gifts, start=1):
                        print(f"{i}. {g}")
                    gchoice = input("输入礼物编号或名称：").strip()
                    gift_name = None
                    if gchoice.isdigit():
                        gi = int(gchoice) - 1
                        if 0 <= gi < len(gifts):
                            gift_name = gifts[gi]
                    else:
                        if gchoice in GIFT_EFFECTS:
                            gift_name = gchoice

                    if gift_name is None:
                        print("无效的礼物选择，未送出任何礼物。")
                    else:
                        self.current_target.give_gift(gift_name)

            elif choice == "3":
                if self.current_target is None:
                    print("当前没有角色。")
                else:
                    print(f"{self.current_target.name} 当前好感度：{self.current_target.affinity}")

            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            if self.current_target and self.current_target.check_ending():
                # 可以在这里放置结局触发的额外描述
                print(f"你与{self.current_target.name}的羁绊达到了顶点，故事线达成。\n")
                break


# 对话库
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


# 礼物影响表（参考需求）
GIFT_EFFECTS = {
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"default": -10},
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}


if __name__ == "__main__":
    game = Game()
    game.start()
