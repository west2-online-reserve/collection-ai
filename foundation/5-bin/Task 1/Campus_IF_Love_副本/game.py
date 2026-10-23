import random
import sys
from typing import Dict, Any

from story import DIALOGUES, GIFT_EFFECTS
from save_manager import load_game, save_game


class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity

    def talk(self) -> int:
        print(f"\n你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogue = DIALOGUES.get(self.name, [])
        if dialogue:
            dialogue_index: int
            if self.affinity < 25:
                dialogue_index = 0
            elif self.affinity < 50:
                dialogue_index = 1
            elif self.affinity < 75:
                dialogue_index = 2
            else:
                dialogue_index = 3

            print(dialogue[dialogue_index]["text"])
            print(f"A: {dialogue[dialogue_index]['optionA']}")
            print(f"B: {dialogue[dialogue_index]['optionB']}")

            while True:
                choice = input("请输入选项：").upper()
                if choice == "A":
                    self.change_affinity(10)
                    break
                elif choice == "B":
                    self.change_affinity(-10)
                    break
                else:
                    print("输入不合法，请选择 A 或 B 。")

            return dialogue_index
        return 0

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        default = GIFT_EFFECTS.get(gift, {}).get("default", 0)  # 获取默认好感度变化值
        effect = GIFT_EFFECTS.get(gift, {}).get(self.name, default)
        self.change_affinity(effect)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        """检查是否达到结局条件"""
        if self.affinity >= 200:
            print(f"你和 {self.name} 的故事已经进入了结局线。")
            return True
        elif self.affinity <= -20:
            print(f"你被 {self.name} 讨厌了，故事线已经结束。")
            return True

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
        # 当前剧情位置
        self.story_block: str = "opening"
        self.story_line: int = 0

    def start(self) -> None:
        """游戏启动菜单"""
        while True:
            print("========== 校园 if·恋 ==========")
            print("1. 开始新游戏")
            print("2. 读取普通存档")
            print("3. 读取即时存档")
            print("4. 退出游戏")

            choice = input("请输入选项：")

            if choice == "1":
                self.start_new_game()
                break
            elif choice == "2":
                if self.load_nomal_save():
                    if not self.resume_game():
                        continue
                    break
            elif choice == "3":
                if self.load_auto_save():
                    if not self.resume_game():
                        continue
                    break
            elif choice == "4":
                print("退出游戏。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

    def start_new_game(self) -> None:
        """开始新游戏"""
        self.story_block = "opening"
        self.story_line = 0
        self.current_target = self.characters["None"]
        for character in self.characters.values():  # 遍历所有角色，重置好感度
            character.affinity = 0

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
            self.story_block = "学姐"
            self.story_line = 0
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
            self.story_block = "小白"
            self.story_line = 0
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
            self.story_block = "姐姐"
            self.story_line = 0
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
            print("4. 保存游戏")
            print("5. 离开（退出游戏）")

            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作

            # 输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.story_block = self.current_target.name  # 更新故事块
                self.story_line = self.current_target.talk()  # 更新故事线索

            # 输入2----
            elif choice == "2":
                print("你可以送的礼物有：")
                for gift in GIFT_EFFECTS.keys():
                    print(f"- {gift}")

                while True:
                    gift_choice = input("请输入你要送的礼物：")
                    if gift_choice in GIFT_EFFECTS:
                        self.current_target.give_gift(gift_choice)
                        self.story_block = gift_choice  # 更新故事块
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
                self.normal_save()

            # 输入5----
            elif choice == "5":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            # 在每次操作后检查好感度是否达到结局条件。
            if self.current_target.affinity <= -20:
                print(f"你被 {self.current_target.name} 讨厌了，故事线结束。")
                self.exit_game()  # 退出游戏
            elif self.current_target.affinity < 0:
                print(f"{self.current_target.name} 对你有些不满，请多加注意。")
            elif self.current_target.affinity >= 90:
                print(f"\n你和 {self.current_target.name} 的故事进入了关键节点！")
                print(f"\n你正在和{self.current_target.name}对话...")
                dialogue = DIALOGUES.get(self.current_target.name, [])
                if dialogue:
                    dialogue_index = 4
                    print(dialogue[dialogue_index]["text"])
                    print(f"A: {dialogue[dialogue_index]['optionA']}")
                    print(f"B: {dialogue[dialogue_index]['optionB']}")
                    while True:
                        choice = input("请输入选项：").upper()
                        if choice == "A":
                            self.current_target.affinity = 200
                            print(
                                f"你成功get到 {self.current_target.name} 的心意，{self.current_target.name} 也对你心生好感。"
                            )
                            print(
                                f"恭喜！你和 {self.current_target.name} 的故事进入了结局线！"
                            )
                            self.exit_game()  # 退出游戏
                            break
                        elif choice == "B":
                            self.current_target.affinity = 80
                            print(
                                f"你没有get到 {self.current_target.name} 的心意，{self.current_target.name} 显得有些失落。"
                            )
                            print(f"当前好感度：{self.current_target.affinity}")
                            break
                        else:
                            print("输入不合法，请选择 A 或 B 。")

            self.auto_save()  # 每次操作后自动保存游戏状态

            if (
                self.current_target == self.characters["None"]
                or self.current_target.check_ending()
            ):
                self.auto_save()
                break

    def exit_game(self) -> None:
        """退出游戏"""
        while True:
            print("\n你要做什么")
            print("1. 保存游戏")
            print("2. 离开（退出游戏）")
            while True:
                choice = input("请输入选项：")
                if choice == "1":
                    self.normal_save()
                    break
                elif choice == "2":
                    print("你选择离开，游戏结束。")
                    self.auto_save()  # 自动保存游戏状态
                    sys.exit(0)
                else:
                    print("无效输入，请重新选择。")
            if choice == "2":
                break

    def create_save_data(self) -> Dict[str, Any]:
        """把当前游戏状态转换成json"""
        return {
            "story_block": self.story_block,
            "story_line": self.story_line,
            "current_target": self.current_target.name,
            "affinity": self.current_target.affinity,
        }

    def normal_save(self) -> None:
        """保存到普通存档"""
        data: Dict[str, Any] = self.create_save_data()
        save_game(data, auto=False)

    def auto_save(self) -> None:
        """保存到即时存档"""
        data: Dict[str, Any] = self.create_save_data()
        save_game(data, auto=True)

    def apply_save_data(self, data: Dict[str, Any]) -> bool:
        """使用存档数据覆盖当前游戏状态"""
        if not data or not isinstance(data, dict):
            return False

        target_name = data.get("current_target")
        story_block = data.get("story_block")
        story_line = data.get("story_line")
        affinity = data.get("affinity")

        # 校验字符串字段以及角色是否存在
        if (
            not isinstance(target_name, str)
            or target_name not in self.characters
            or not isinstance(story_block, str)
        ):
            print("存档角色或剧情块不正确。")
            return False

        # bool 是 int 的子类，使用 type(...) is int 排除 True、False
        if type(story_line) is not int or type(affinity) is not int:
            print("剧情索引和好感度必须是整数。")
            return False

        if story_line < 0:
            print("剧情索引不能为负数。")
            return False

        if story_block in DIALOGUES:
            dialogues = DIALOGUES[story_block]

            if story_block != target_name:
                print("剧情角色与当前角色不一致。")
                return False

            if story_line >= len(dialogues):
                print("剧情索引超出范围。")
                return False

        elif story_block != "opening" and story_block not in GIFT_EFFECTS:
            print("存档中的剧情块不存在。")
            return False

        self.story_block = story_block
        self.story_line = story_line
        self.current_target = self.characters[target_name]
        self.current_target.affinity = affinity
        return True

    def load_nomal_save(self) -> bool:
        """加载普通存档"""
        data: Dict[str, Any] = load_game(auto=False)

        if self.apply_save_data(data):
            print("普通存档加载成功。")
            return True

        return False

    def load_auto_save(self) -> bool:
        """加载即时存档"""
        data: Dict[str, Any] = load_game(auto=True)

        if self.apply_save_data(data):
            print("即时存档加载成功。")
            return True

        return False

    def resume_game(self) -> bool:
        """继续游戏"""
        if self.current_target == self.characters["None"]:
            print("当前没有进行中的故事线，请开始新游戏。\n")
            return False

        if self.current_target.check_ending():
            print("当前故事线已结束，请开始新游戏。\n")
            return False

        if self.story_block == "opening":
            print("当前处于开场剧情，请开始新游戏。\n")
            return False
        elif self.story_block in GIFT_EFFECTS:
            print(f"当前处于送礼物剧情，最近送的礼物是：{self.story_block}。\n")
        else:
            print(f"当前处于对话剧情，最近对话的角色是：{self.story_block}。")
            print(
                f"最近对话的内容是：{DIALOGUES.get(self.story_block, [{}])[self.story_line].get("text", "")}\n"
            )

        print("========== 继续游戏 ==========")
        print(
            f"当前目标角色：{self.current_target.name}，好感度：{self.current_target.affinity}"
        )
        self.story_loop()
        return True
