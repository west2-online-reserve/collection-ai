import sys, json, os
from typing import Any
from data import DIALOGUES, GIFT_EFFECTS, zhonglei
from character import Character

path: str = os.path.dirname(os.path.abspath(__file__))
re_path: str = os.path.join(path, "save.json")


class Game:
    def __init__(self) -> None:
        self.characters: dict[str, Character] = {
            "Touma Kazusa": Character("Touma Kazusa", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target: Character = self.characters["None"]
        self.save_count: int = 0

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # Touma Kazusa场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def save(self) -> None:
        save_data: dict[str, str | int] = {
            "save_characters": self.current_target.name,
            "save_affinity": self.current_target.affinity,
            "save_count": self.current_target.count,
        }
        with open(re_path, "r", encoding="utf-8") as f:
            saves: dict[str, Any] = json.load(f)
        if len(saves["man"]) == 10:
            num = input("存档已满，请选择要覆盖的存档（1-10）：\n或输入返回\n")
        else:
            num = input("请选择要保存的存档位置（1-10）：\n或输入返回\n")
        if num == "返回":
            return
        try:
            num = int(num)
            if num < 1 or num > 10:
                print("请输入正确的数字")
                return
        except ValueError:
            print("请输入数字")
            return
        saves["man"][str(num)] = save_data
        with open(re_path, "w", encoding="utf-8") as f:
            json.dump(saves, f, ensure_ascii=False)
        print(f"成功保存到了{num}号存档")

    def load(self, kind: str) -> None:
        try:
            with open(re_path, "r", encoding="utf-8") as f:
                saves: dict[str, Any] = json.load(f)
        except FileNotFoundError:
            print("无存档")
            return

        if kind == "man":
            for i in saves["man"]:
                print(
                    f"手动存档：\n{i}: 角色：{saves['man'][i]['save_characters']},对话次数：{saves['man'][i]['save_count']},好感度：{saves['man'][i]['save_affinity']}"
                )
            num = input("请输入想要的存档序号（1-10）：\n或输入返回\n")
        elif kind == "auto":
            for i in saves["auto"]:
                print(
                    f"自动存档：\n{i}角色:{saves['auto'][i]['save_characters']},对话次数：{saves['auto'][i]['save_count']},好感度：{saves['auto'][i]['save_affinity']}"
                )
            num = input("请输入想要的存档序号（1-5）：\n或输入返回\n")
        else:
            print("请输入正确的种类")
            return
        if num == "返回":
            return
        try:
            load_data = saves[kind][num]
        except KeyError:
            print("请输入正确的序号")
            return
        self.current_target = self.characters[load_data["save_characters"]]
        self.current_target.affinity = load_data["save_affinity"]
        self.current_target.count = load_data["save_count"]
        print("读档成功")

    def auto_save(self) -> None:
        save_data: dict[str, str | int] = {
            "save_characters": self.current_target.name,
            "save_affinity": self.current_target.affinity,
            "save_count": self.current_target.count,
        }
        try:
            with open(re_path, "r", encoding="utf-8") as f:
                auto_saves: dict[str, Any] = json.load(f)
        except FileNotFoundError:
            auto_saves = {"auto": {}, "man": {}, "存档计数": 1}
        self.save_count = auto_saves["存档计数"]
        if self.save_count == 6:
            self.save_count = 1
        auto_saves["auto"][str(self.save_count)] = save_data
        auto_saves["存档计数"] = self.save_count
        with open(re_path, "w", encoding="utf-8") as f:
            json.dump(auto_saves, f, ensure_ascii=False)
        self.save_count += 1

    def scene_senpai(self) -> bool:
        print("\n【场景一：社团Touma Kazusa】")
        print("你路过社团活动室，Touma Kazusa正拿着画板注意到你。")
        print("Touma Kazusa：『这位新生？要不要来试试？』")

        choice: str = input(
            "1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择："
        )
        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("Touma Kazusa目光一震，眼神变得格外认真。你进入【Touma Kazusa线】！")
            self.current_target = self.characters["Touma Kazusa"]
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice: str = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print(
                """\n你探头看了看，顺口报出来答案。小白疑惑地转过来，眼神在扫过你精致的面容时停下来。
            ‘你谁啊？’
            ‘诶多，我是xxx。不过这不重要。’
            说完，你自顾自地凑上去，拿过笔，在纸上写出完美的解法，顺带把重难点都标红了。"""
            )
            print(
                "小白因为你的靠近红了脸，但眼神里又充满了对知识的渴望，于是拉着你要求为她讲解别的题目。你进入【小白线】！"
            )
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("""‘老弟啥废物啊这都不会。’
            小白一脸疑惑并带着看臭虫的表情转过身来。
            ‘有病！’
            随后戴上她的大号耳机，专注地学习。""")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐")
        print(
            "你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来..."
        )
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice: str = input(
            "1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择："
        )
        if choice == "1":
            print("你游刃有余地讲出通俗易懂的解释，姐姐明显被你巧妙的思路震住了。")
            print(
                "‘挺有趣的，下次我请你喝咖啡吧。当作你提供思路的谢礼了。’你进入【姐姐线】"
            )
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("""你继续敲代码，无视知性姐姐的搭理。
            那位穿着风衣的姐姐也就知趣地离开了。
            ‘无聊。’""")
            return False

    def story_loop(self) -> None:
        """角色线主循环"""
        while True:
            self.auto_save()
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")
            print("5. 保存当前进度")
            print("6. 读取游戏进度")
            choice: str = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作

            match choice:
                case "1":
                    a: bool = self.current_target.talk()
                    if not a:
                        print(
                            f"game over。\n {self.current_target.name}对你的耐心消磨殆尽了。\n 不是哥们，多送几次礼物的事，这么多次机会都拿不下吗？？？"
                        )
                        return
                    else:

                        cho: str = input("请输入a或b：")
                        cho = "option" + cho.upper()
                        try:
                            print(
                                DIALOGUES[self.current_target.name][
                                    self.current_target.count - 1
                                ][cho][0]
                            )
                            self.current_target.change_affinity(
                                DIALOGUES[self.current_target.name][
                                    self.current_target.count - 1
                                ][cho][1]
                            )
                        except KeyError:
                            print("只能输入a或b")

                case "2":
                    print(
                        "礼物：鲜花，编程笔记，奶茶，奇怪的石头，精致的钢笔，可爱玩偶，夜宵外卖\n或返回\n"
                    )
                    gift: str = input("你想送出什么礼物：")

                    if gift == "返回":
                        continue

                    if gift not in GIFT_EFFECTS:
                        print("你的财力只允许送出这些礼物")
                    else:
                        self.current_target.give_gift(gift)

                case "3":
                    print(
                        f"{self.current_target.name}的好感度为：{self.current_target.affinity}"
                    )

                case "4":
                    print("你选择离开，游戏结束。")
                    sys.exit(0)

                case "5":
                    self.save()

                case "6":
                    kind: str = input("""请输入想要读取哪种存档
                                  1.自动存档
                                  2.手动存档
                                            """)
                    if not (kind == "1" or kind == "2"):
                        print("请输入1或2")
                        continue
                    kind = zhonglei[kind]
                    self.load(kind)

                case _:
                    print("无效输入，请重新选择。")

            if (
                self.current_target == self.characters["None"]
                or self.current_target.check_ending()
            ):
                break


if __name__ == "__main__":
    game: Game = Game()
    game.start()
