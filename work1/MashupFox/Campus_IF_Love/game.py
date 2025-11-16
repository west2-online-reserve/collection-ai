import sys
from typing import Dict, Optional
from character import Character
from data import DIALOGUES, GIFT_EFFECTS, AVAILABLE_GIFTS
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
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于'神秘新生代表'的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        if not self.scene_senpai():
            if not self.scene_xiaobai():
                if not self.scene_jiejie():
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
            print("在众目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print("\n你轻松解决了难题，小白眼中闪烁着崇拜的光芒。")
            print("小白：『你好厉害啊！能教教我吗？』你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("你敷衍了几句，匆匆离开了图书馆。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你详细解释了代码逻辑，姐姐听得十分认真。")
            print("姐姐：『你的思维方式很独特呢……』你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你继续埋头敲代码，姐姐略显失望地离开了。")
            return False

    def story_loop(self) -> None:
        """角色线主循环"""
        if self.current_target is None:
            return

        while True:
            print(f"\n=== {self.current_target.name}线 ===")
            print("你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：")

            if choice == "1":
                
                dialogues = DIALOGUES.get(self.current_target.name, [])
                self.current_target.talk(dialogues)
                
            elif choice == "2":
                
                print("\n可选的礼物：")
                for i, gift in enumerate(AVAILABLE_GIFTS, 1):
                    print(f"{i}. {gift}")
                
                try:
                    gift_choice = int(input("请选择礼物编号：")) - 1
                    if 0 <= gift_choice < len(AVAILABLE_GIFTS):
                        gift = AVAILABLE_GIFTS[gift_choice]
                        self.current_target.give_gift(gift, GIFT_EFFECTS)
                    else:
                        print("无效的礼物选择！")
                except ValueError:
                    print("请输入有效的数字！")
                    
            elif choice == "3":
                
                print(f"\n当前好感度：{self.current_target.affinity}")
                if self.current_target.affinity < 30:
                    print("关系：陌生人")
                elif self.current_target.affinity < 60:
                    print("关系：朋友")
                elif self.current_target.affinity < 100:
                    print("关系：好朋友")
                else:
                    print("关系：特别的人")
                    
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                print("\n游戏通关！感谢游玩！")
                break

if __name__ == "__main__": 
    game = Game()
    game.start()