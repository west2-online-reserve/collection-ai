import sys
from typing import Optional,List,Dict
from character import Character
from constant import DIALOGUES, GIFT_EFFECTS

class Game:
    def __init__(self) -> None:
        self.characters: Dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target: Optional[Character] = None  # 当前互动角色

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入场景，若都不选则触发“吐槽”
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self) -> bool:
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice: str = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
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

        choice: str = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print("\n你给她讲解了思路，并写出伪代码，小白的眼睛亮了起来。")
            print("小白：『太感谢你了！我明白了！』")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看着你离开，眼中闪过一丝失落。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice: str = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你给她讲了代码的关键点，姐姐微微一笑，和你坐在一起")
            print("姐姐：『我们一起讨论吧』")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("姐姐转身离开了。")
            return False
        
    def story_loop(self) -> None:
        """角色线主循环：处理“聊天、送礼物、查好感、退出”的交互"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice: str = input("请输入选项：")

            if choice == "1":
                self.current_target.talk()
            elif choice == "2":
                print("可选礼物：鲜花/编程笔记/奶茶/奇怪的石头/精致的钢笔/可爱玩偶/夜宵外卖")
                gift: str = input("请输入礼物名称：").strip()
                self.current_target.give_gift(gift)
            elif choice == "3":
                print(f"{self.current_target.name} 当前好感度：{self.current_target.affinity}")
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            # 检查是否触发结局
            if self.current_target.check_ending():
                break

if __name__ == "__main__": 
    game = Game()
    game.start()