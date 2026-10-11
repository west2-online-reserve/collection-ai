import typing
import sys
from .character import Character
from .config import *
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent / ".." / "saves"


def confirm(action: typing.Literal["退出游戏", "重新开始游戏", "存档", "读档"]) -> bool:
    print(confirm_text.format(action=action))
    confirm_choice = input(confirm_option)
    if confirm_choice == "1":
        return True
    elif confirm_choice == "2":
        return False
    else:
        print(wrong_output["None"])
        return confirm(action=action)  # 如果输入错误 则重新调用函数


def save(
    wife: typing.Literal["self.current_target"],
    affinity: typing.Literal["self.affinity"],
    talk_times: typing.Literal["self.talk_times"],
    save_method: typing.Literal["save", "Qsave"] = "save",
) -> None:
    if save_method == "save":
        save_path = BASE / "save.json"

    else:
        save_path = BASE / "Qsave.json"

    data = {
        "wife": wife,
        "affinity": affinity,
        "talk_times": talk_times,
    }
    with open(save_path, "w+", encoding="utf-8") as save:
        json.dump(data, save, ensure_ascii=False, indent=4)
    if save_method == "save":
        print(save_finish.format(path=(save_path.resolve())))


def load(load_method: typing.Literal["load", "Qload"] = "load") -> dict:
    if load_method == "load":
        load_path = BASE / "save.json"

    else:
        load_path = BASE / "Qsave.json"

    with open(load_path, "r+", encoding="utf-8") as load:
        data = json.load(load)
    if load_method == "load":
        print(load_finish.format(path=(load_path.resolve())))
    return data


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character(
                "学姐", "社团里的艺术少女"
            ),  # 当索引键的值时直接实例化对象
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target = self.characters["None"]

    def start(self) -> None:
        for i in welcome:
            print(i)
        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print(xianchong)

    def scene_senpai(self) -> bool:
        print(scene_title[0])
        print(scene_meet[0])
        print(scene_talk[0])
        while True:
            scene_choice = input(scene_option[0])
            if scene_choice == "1":
                print(scene_accept[0])
                print(scene_reflect[0])
                self.current_target = self.characters["学姐"]  # 这里直接实例化对象
                # 等价于self.current_target=Character("学姐")
                self.story_loop()
                return True
            elif scene_choice == "2":
                print(scene_refuse[0])
                return False
            else:
                print(wrong_output["学姐"])
                print(wrong_output_tip)
        # 这里加了一个无效输入的判定

    def scene_xiaobai(self) -> bool:
        print(scene_title[1])
        print(scene_meet[1])
        print(scene_talk[1])
        while True:
            scene_choice = input(scene_option[1])
            if scene_choice == "1":
                print(scene_accept[1])
                print(scene_reflect[1])
                self.current_target = self.characters["小白"]
                self.phrase = 1
                self.story_loop()
                return True
            elif scene_choice == "2":
                print(scene_refuse[1])
                return False
            else:
                print(wrong_output["小白"])
                print(wrong_output_tip)
        # scene_choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了 1 时 输出了 print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线

    def scene_jiejie(self) -> bool:
        print(scene_title[2])
        print(scene_meet[2])
        print(scene_talk[2])

        # choice = input(
        #    "1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择："
        # )
        while True:
            scene_choice = input(scene_option[2])
            if scene_choice == "1":
                print(scene_accept[2])
                print(scene_reflect[2])
                self.current_target = self.characters["姐姐"]
                self.phrase = 1
                self.story_loop()
                return True
            elif scene_choice == "2":
                print(scene_refuse[2])
                return False
            else:
                print(wrong_output["姐姐"])
                print(wrong_output_tip)
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        # 要求同上

    def story_loop(self) -> None:
        """角色线主循环"""
        while True:
            save(
                wife=self.current_target.name,
                affinity=self.current_target.affinity,
                talk_times=self.current_target.talk_times,
                save_method="Qsave",
            )

            for i in menu:
                print(i)

            choice = input(loop_option)

            # TODO 完成输入不同选项时 进行的操作
            if choice == "1":
                self.current_target.talk()

            # 输入 1---关于聊天的内容可以自己构思 也可以从剧本中截取
            elif choice == "2":
                self.current_target.give_gift()

            # 输入 2----\
            elif choice == "3":
                print(
                    check_affinity.format(
                        self=self.current_target, affinity=self.current_target.affinity
                    )
                )

            # 输入 3----

            elif choice == "4":  # 存档
                if confirm(choice_text[4]):
                    # from start import save
                    save(
                        wife=self.current_target.name,
                        affinity=self.current_target.affinity,
                        talk_times=self.current_target.talk_times,
                    )
            elif choice == "5":  # 读档
                if confirm(choice_text[5]):
                    # from start import load
                    try:
                        data = load()
                        self.current_target = self.characters[data["wife"]]
                        self.current_target.affinity = data["affinity"]
                        self.current_target.talk_times = data["talk_times"]
                    except FileNotFoundError:
                        print(file_error)
            elif choice == "6":  # 离开
                if confirm(choice_text[6]):
                    print(leave_text)
                    sys.exit(0)
            elif choice == "7":  # 重开
                print(restart)
                self.reset()
                self.start()

            elif choice == "24":
                print(caidan_24)
            else:
                print(wrong_output["None"])

            if (
                self.current_target
                == self.characters[
                    "None"
                ]  # 读取存档使要赋值self.characters,不然每次读取存档后都会跳回主菜单
                or self.current_target.check_ending()
            ):
                while True:
                    print(ending_tips)
                    for option in ending_options:
                        print(option)
                    end_choice = input(loop_option)
                    if end_choice == "1":  # 重新开始游戏
                        print(restart)
                        self.reset()
                        self.start()

                    elif end_choice == "2":  # 退出至菜单
                        return None
                    elif end_choice == "3":  # 退出游戏
                        print(leave_text)
                        sys.exit(0)
                    else:
                        print(wrong_output["None"])

    def reset(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target = self.characters["None"]


if __name__ == "__main__":
    game = Game()
    game.start()
