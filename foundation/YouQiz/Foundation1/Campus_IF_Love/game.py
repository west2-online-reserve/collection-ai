import json
import sys
from typing import Literal, Dict
from character import Character
from story import DIALOGUES, GIFT_EFFECTS

AUTO_SAVE_PATH = "autosave.json"
SAVE1_PATH = "save1.json"

class Game:
    def __init__(self) -> None:
        self.characters: Dict = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target: Character = self.characters["None"]
        self.scene_code: int = 0  # 剧情块编码:0未选择角色，1学姐线，2小白线，3姐姐线

    def auto_save(self) -> None:
        """即时存档：每次操作自动写入autosave.json"""
        save_data = {
            "scene_code": self.scene_code,
            "current_target": self.current_target.to_save_dict()
        }
        with open(AUTO_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def manual_save(self) -> None:
        """手动存档1号槽"""
        save_data = {
            "scene_code": self.scene_code,
            "current_target": self.current_target.to_save_dict()
        }
        with open(SAVE1_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print("手动存档成功（save1.json）")

    def load_save(self, filepath: str) -> bool:
        """读档，返回是否成功"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.scene_code = data["scene_code"]
            char_data = data["current_target"]
            self.current_target = Character.from_save_dict(char_data)
            print(f"读档成功，当前角色：{self.current_target.name}，好感：{self.current_target.affinity}")
            return True
        except FileNotFoundError:
            print("存档文件不存在！")
            return False
        except Exception as e:
            print(f"读档失败：{e}")
            return False

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

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.scene_code = 1
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
            print("\n你随手挑起一只笔，主动帮她解题。")
            print("小白目光一震，眼神变得格外认真。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.scene_code = 2
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_jiejie(self) ->bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 要求同上
        if choice == "1":
            print("\n你缓缓低眉，毫不在意的开始解释。")
            print("姐姐目光一震，眼神变得格外认真。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.scene_code = 3
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def story_loop(self) -> None:
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 手动存档(save1.json)")
            print("5. 读取存档(save1.json)")
            print("6. 离开（退出游戏）")

            choice = input("请输入选项：").strip()

            # TODO 完成输入不同选项时 进行的操作

            # 输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.current_target.talk()
                self.auto_save()

            # 输入2----
            elif choice == "2":
                gift_names = ", ".join(GIFT_EFFECTS.keys())
                gift = input(f"你想送的礼物是什么，可选一个：{gift_names}\n").strip()
                while gift not in GIFT_EFFECTS:
                    gift = input(f"礼物无效！可选礼物：{gift_names}\n").strip()
                self.current_target.give_gift(gift)
                self.auto_save()

            # 输入3----
            elif choice == "3":
                print(f"当前{self.current_target.name}的好感度为：{self.current_target.affinity}")
            elif choice == "4":
                self.manual_save()
            elif choice == "5":
                self.load_save(SAVE1_PATH)
            elif choice == "6":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            if self.current_target == self.characters["None"] or self.current_target.check_ending():
                break