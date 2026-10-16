import sys
import json
from character import Character
from data import DIALOGUES, GIFT_EFFECTS

class Game:
    def __init__(self) -> None:
        self.characters: dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target: Character = self.characters["None"]
        self.save_file: str = "game_save.json"
        self.autosave_file: str = "game_autosave.json"

    def save_game(self, file: str) -> None:
        data: dict ={
            "current_target": self.current_target.name,
            "current_affinity": self.current_target.affinity,
            "current_cnt": self.current_target.cnt
        }
        with open(file, "w", encoding="utf-8")as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("游戏已保存。")

    def load_game(self, file: str) -> bool:
        try:
            with open(file, "r", encoding="utf-8")as f:
                data: dict = json.load(f)
                self.current_target = self.characters[data["current_target"]]
                self.current_target.affinity = data["current_affinity"]
                self.current_target.cnt = data["current_cnt"]
                print("游戏已加载。")
                return True
                
        except FileNotFoundError:
            print("存档文件未找到。")
            return False

    def start(self) -> None:
        try:
            with open(self.autosave_file, "r", encoding="utf-8")as f:
                pass
            print("检测到自动存档文件，是否加载？(y/n)")
            choice: str = input()
            if choice == "y":
                self.load_game(self.autosave_file)
                self.story_loop()
        except FileNotFoundError:
            print("自动存档文件未找到。")
            pass
            

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
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        #注意 除了判断外 你可以同时输出角色的反应 
        #比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        #写法可以借鉴学姐线
        if choice == "1":
            print("\n你耐心地为小白讲解算法题的解法，她的眼睛闪烁着光芒，感激地看着你。")
            print("小白：『谢谢你！你真是太厉害了！』")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看着你离去的背影，眼中闪过一丝失落。")
            return False
        

            


    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice: str = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        #要求同上
        if choice == "1":
            print("\n你耐心地向姐姐解释你的代码实现方法，她的眼神中闪过一丝赞赏。")
            print("姐姐：『原来如此，你的思路很清晰呢。』")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("姐姐看着你专注敲代码的样子，微微皱眉，似乎有些失望。")
            return False
        
    def story_loop(self) -> None:
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 保存游戏")
            print("5. 读取存档")
            print("6. 离开")

            choice: str = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作 

            #输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.current_target.talk()



            #输入2----
            elif choice == "2":
                print("你可以送出的礼物如下：")
                for i in range(len(GIFT_EFFECTS)):
                    print(f"{i+1}.{list(GIFT_EFFECTS.keys())[i]}")
                gift: str = input("请输入你要送出的礼物：")
                self.current_target.give_gift(list(GIFT_EFFECTS.keys())[int(gift)-1])


            #输入3----
            elif choice == "3":
                print(f"{self.current_target.name} 的当前好感度为：{self.current_target.affinity}")


            elif choice == "4":
                self.save_game(self.save_file)
                continue


            elif choice == "5":
                if self.load_game(self.save_file):
                    print("存档已读取。")
                else:
                    print("存档读取失败。")
                continue


            elif choice == "6":
                print("你选择离开，游戏结束。")
                sys.exit(0)

         
            else:
                print("无效输入，请重新选择。")

            if self.current_target == self.characters["None"] or self.current_target.check_ending():
                break

            self.save_game(self.autosave_file)

if __name__ == "__main__": 
    game: Game = Game()
    game.start()