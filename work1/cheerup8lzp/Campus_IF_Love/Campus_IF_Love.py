import sys
from typing import Dict, List, Optional, Any


class Character:
    name: str
    role: str
    affinity: int

    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        # 从对话库里获取对话列表
        l_senpai: List[Dict[str, str]] = DIALOGUES.get(self.name, [])
        for dialogue in l_senpai:
            print(f"{self.name}：『{dialogue['text']}』")
            print("1.", dialogue["optionA"])
            print("2.", dialogue["optionB"])
            choice = input("请选择：")
            if choice == "1":
                print(f"\n你选择了：{dialogue['optionA']}")
                self.change_affinity(10)
            elif choice == "2":
                print(f"\n你选择了：{dialogue['optionB']}")
        # 聊天结束后略微提升好感度
        self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        effects: Dict[str, int] = GIFT_EFFECTS.get(gift, {})
        # 优先查找针对该角色的效果，否则查找 default
        value = effects.get(self.name, effects.get("default", 0))
        self.change_affinity(int(value))

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


class Game:
    characters: Dict[str, Character]
    current_target: Optional[Character]

    def __init__(self) -> None:
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
        }
        self.current_target = None

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景，如果都不选择则结束
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
            print("\n你耐心地为小白讲解算法思路，她的眼睛顿时亮了起来。")
            print("小白：『哇，谢谢你！你真是太厉害了！』你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看着你离去的背影，失落地叹了口气。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你详细地向姐姐解释了你的代码实现，她露出了欣赏的笑容。")
            print("姐姐：『真是个有才华的年轻人呢！』你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("姐姐看着你专注敲代码的样子，微微一笑，转身离开了咖啡店。")
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
                if self.current_target:
                    self.current_target.talk()
                else:
                    print("当前没有目标角色。")
            elif choice == "2":
                print("你有什么礼物想送给她？")
                print("1. 鲜花")
                print("2. 编程笔记")
                print("3. 奶茶")
                print("4. 奇怪的石头")
                print("5. 精致的钢笔")
                print("6. 可爱玩偶")
                print("7. 夜宵外卖")

                gift_choice = input("请选择礼物：")
                gifts: Dict[str, str] = {
                    "1": "鲜花",
                    "2": "编程笔记",
                    "3": "奶茶",
                    "4": "奇怪的石头",
                    "5": "精致的钢笔",
                    "6": "可爱玩偶",
                    "7": "夜宵外卖",
                }

                gift = gifts.get(gift_choice)
                if gift:
                    if self.current_target:
                        self.current_target.give_gift(gift)
                else:
                    print("无效的礼物选择。")
            elif choice == "3":
                if self.current_target:
                    print(f"{self.current_target.name} 的当前好感度：{self.current_target.affinity}")
                else:
                    print("当前没有目标角色。")
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            if self.current_target and self.current_target.check_ending():
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
