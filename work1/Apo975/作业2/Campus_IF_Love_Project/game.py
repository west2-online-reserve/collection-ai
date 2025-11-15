import sys
import time
from typing import Dict, Optional, List, Any
from character import Character
from story import DIALOGUES, GIFT_EFFECTS
from save import save, load

def prints(text: str, delay: float = 0.06) -> None:
    for char in text:
        print(char, end='', flush=True)  # 逐字输出
        time.sleep(delay)
    print()

class Game:
    def __init__(self) -> None:
        # 初始化角色数据
        self.characters: Dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target: Optional[Character] = None  # 当前互动角色
        self.scene_stage: str = "initial"  # 剧情进度

    def get_save_data(self) -> Dict[str, Any]:
        # 生成存档数据
        return {
            # 角色属性
            "characters": {
                name: {
                    "affinity": char.affinity,  # 好感度
                    "role": char.role           # 身份
                } for name, char in self.characters.items()
            },
            "current_target": self.current_target.name if self.current_target else None,  # 当前目标角色
            "scene_stage": self.scene_stage  # 剧情进度
        }

    def load_save_data(self, data: Dict[str, Any]) -> None:
        #继承存档数据游戏状态
        # 恢复角色属性
        for name, char_data in data["characters"].items():
            self.characters[name].affinity = char_data["affinity"]
            self.characters[name].role = char_data["role"]
        # 恢复当前互动角色
        target_name = data["current_target"]
        self.current_target = self.characters[target_name] if target_name else None
        # 恢复剧情进度
        self.scene_stage = data["scene_stage"]
        print("读档成功，已恢复游戏状态")

    def _auto_save(self) -> None:
        # 即时存档
        save(self.get_save_data(), is_quick=True)

    def start(self) -> None:
        """游戏入口：启动剧情流程"""
        print("========== 游戏开始：校园 if·恋 ==========")
        prints("你是一名刚刚踏入大学校园的新生。")
        prints("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        prints("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        prints("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        prints("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")
        time.sleep(1)

        self.scene_stage = "scene_senpai"  # 更新剧情阶段
        self._auto_save()  # 初始剧情后存档

        # 依次进入三个场景
        if not self.scene_senpai():
            time.sleep(1)
            self.scene_stage = "scene_xiaobai"
            self._auto_save()
            if not self.scene_xiaobai():
                time.sleep(1)
                self.scene_stage = "scene_jiejie"
                self._auto_save()
                if not self.scene_jiejie():
                    print("\n啥，眼前三妹子都不要？？死现充别玩旮旯给木！！！")

    def scene_senpai(self) -> bool:
        """学姐场景剧情"""
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")
        choice: str = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：").strip()

        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.scene_stage = "senpai_line"
            self._auto_save()
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            self._auto_save()
            return False

    def scene_xiaobai(self) -> bool:
        """小白场景剧情"""
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice: str = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：").strip()

        if choice == "1":
            print("\n你使用你高超的代码技术，一下子就给出了这道题的最优解")
            print("小白眼冒金光，满眼都是对你的崇拜。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.scene_stage = "xiaobai_line"
            self._auto_save()
            self.story_loop()
            return True
        else:
            print("你冷漠地走开，留下小白独自苦恼。")
            self._auto_save()
            return False

    def scene_jiejie(self) -> bool:
        """姐姐场景剧情"""
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")
        choice: str = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：").strip()

        if choice == "1":
            print("\n你的双手在键盘上飞舞，一边自如地完善代码，一边给姐姐讲解思路")
            print("姐姐看起来很惊讶，对你投下了异样的目光。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.scene_stage = "jiejie_line"
            self._auto_save()
            self.story_loop()
            return True
        else:
            print("你一言不发，姐姐很失望地离开了。")
            self._auto_save()
            return False

    def story_loop(self) -> None:
        """角色线主循环：处理互动操作"""
        while True:
            time.sleep(0.5)
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 手动存档")
            print("5. 读取存档")
            print("6. 离开（退出游戏）")

            choice: str = input("请输入选项：").strip()

            if choice == "1":
                #聊天
                print("来聊点什么吧！")
                self.current_target.talk()
                self._auto_save()

            elif choice == "2":
                # 送礼
                print("要送什么礼物呢？")
                gift_list: List[str] = list(GIFT_EFFECTS.keys())
                for i, gift in enumerate(gift_list):
                    print(f"{i+1}: {gift}")
                print("请选择： (输入礼物对应的数字)")
                while True:
                    try:
                        ans: int = int(input().strip())
                        if 1 <= ans <= len(gift_list):
                            ans -= 1
                            print(f"你选择了{gift_list[ans]}")
                            break
                        else:
                            print(f"输入错误，请输入1到{len(gift_list)}之间的数字！")
                    except ValueError:
                        print("输入错误，请输入数字！")

                self.current_target.give_gift(gift_list[ans])
                self._auto_save()

            elif choice == "3":
                # 查好感度
                print(f"{self.current_target.name}当前的好感度为：{self.current_target.affinity}")

            elif choice == "4":
                # 手动存档
                save(self.get_save_data(), is_quick=False)

            elif choice == "5":
                # 读取存档
                load_type = input("1. 读取即时存档  2. 读取手动存档：").strip()
                if load_type == "1":
                    data = load(is_quick=True)
                elif load_type == "2":
                    data = load(is_quick=False)
                else:
                    print("无效选择，请重新输入")
                    continue
                if data: #检查有没有正确存档返回
                    self.load_save_data(data)
                    continue

            elif choice == "6":
                # 退出游戏
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                break


if __name__ == "__main__":
    game: Game = Game()
    game.start()