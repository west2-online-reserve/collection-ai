import json
import os
import sys
import random
from character import Character
from data import DIALOGUES, GIFT_EFFECTS, GIRL_END


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target = self.characters["None"]

    def reset_game(self):
        """开启新游戏前，清空所有人的好感度"""
        for char in self.characters.values():
            char.affinity = 0
        self.current_target = self.characters["None"]

    def save_game(self):
        """保存当前角色状态和所有人的好感度"""
        print("这里可以进行存档\n")
        save_num = input("请输入存档编号(1-3)：")
        try:
            save_num = int(save_num)
        except ValueError:
            print("输入无效，需要输入数字")
            return
            
        if save_num < 1 or save_num > 3:
            print("存档编号无效")
            return
            
        filename = f"save{save_num}.json"
        data = {
            "current_target": self.current_target.name,
            "affinities": {
                name: char.affinity for name, char in self.characters.items()
            },
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print("存档成功")

    def load_game(self) -> bool:
        """读取游戏进度"""
        save_num = input("请输入存档编号(1-3)：")
        try:
            save_num = int(save_num)
        except ValueError:
            print("输入无效，需要输入数字")
            return False
            
        if save_num < 1 or save_num > 3:
            print("存档编号无效")
            return False
            
        filename = f"save{save_num}.json"
        if not os.path.exists(filename):
            print("你存了吗你就读？")
            return False
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.current_target = self.characters.get(
                data["current_target"], self.characters["None"]
            )
            for name, affinity in data["affinities"].items():
                if name in self.characters:
                    self.characters[name].affinity = affinity
            if self.current_target.name != "None":
                print("你当前的角色路线是：", self.current_target.name)
                self.story_loop()
            return True

    def del_game(self):
        # 删除存档
        save_num = input("请输入要删除的存档编号(1-3)：")
        try:
            save_num = int(save_num)
        except ValueError:
            print("输入无效，需要输入数字")
            return
            
        if save_num < 1 or save_num > 3:
            print("存档编号无效")
            return
            
        filename = f"save{save_num}.json"
        if os.path.exists(filename):
            choice = input("是否删除存档？（Y/N）")
            if choice.upper() == "Y":
                os.remove(filename)
                print("存档已删除")
        else:
            print("没有存档可以删除")

    def start(self):
        while True:
            print("========== 游戏开始：校园 if·恋 ==========")

            # 增加存档读取功能
            # 优化游戏退出逻辑
            print("1. 开始新游戏")
            print("2. 读取存档")
            print("3. 删除存档")
            print("4. 退出游戏")
            start_choice = input("请选择：")

            if start_choice == "4":
                print("期待与你再见喵！")
                sys.exit()

            if start_choice == "3":
                self.del_game()
                continue
                
            if start_choice == "2":
                if self.load_game():
                    continue
                else:
                    continue

            if start_choice == "1":
                self.reset_game()
            else:
                print("小手不是很老实")
                continue

            print("你是一名刚刚踏入大学校园的新生。")
            skip_intro = input("你可以选择跳过这段前置剧情，该选项仅会出现一次（Y/N）")
            if skip_intro.upper() == "Y":
                print("跳过前置剧情，直接进入角色选择界面！")
                self.choice_girl()
                continue
            print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
            print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
            print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
            print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

            # 依次进入三个场景 如果三个都不选....
            self.choice_girl()

    def choice_girl(self):
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self):
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
            self.story_loop()
            return True
        else:
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线
        if choice == "1":
            print("\n你耐心地坐下来，和小白一起分析题目，逐步解开了她的困惑。")
            print("小白眼睛一亮，对你充满了感激和崇拜。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("你随口说了几句，表示自己也不太清楚，然后转身离开了。")
            return False

    def scene_jiejie(self):
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
            print("\n你抬起头，微微一笑，开始向姐姐解释你的代码实现方法。")
            print("姐姐听得津津有味，对你的才华和自信印象深刻。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你继续专注地敲代码，完全没有注意到姐姐的存在。")
            return False

    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 存档")
            print("5. 离开（退出游戏）")
            if self.current_target.affinity >= 100:
                print(
                    f"恭喜你，你与{self.current_target.name}的好感度已经很高了！可以进入结局了！"
                )
                print("0. 迈入结局")
            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作

            # 输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.current_target.talk()

            # 输入2----
            elif choice == "2":
                while True:
                    gift = input(
                        "你想送什么礼物？（鲜花/编程笔记/奶茶/奇怪的石头/精致的钢笔/可爱玩偶/夜宵外卖）"
                    )
                    if gift in GIFT_EFFECTS:
                        self.current_target.give_gift(gift)
                        break
                    else:
                        print("没有这个东西的喵")
            # 输入3----
            elif choice == "3":
                print(
                    f"你当前位于{self.current_target.name}线，好感度为{self.current_target.affinity}"
                )

            elif choice == "4":
                self.save_game()

            elif choice == "5":
                user_input = input("你真的要离开吗？（Y/N）")
                if user_input.upper() == "Y":
                    print("期待与你再见喵！")
                    return
                else:
                    print("继续游戏！")

            elif choice == "0":
                if self.current_target.affinity < 100:
                    print("无效输入，请重新选择。")
                else:
                    print(f"恭喜！你和 {self.current_target.name} 的故事进入了结局线！")
                    print(f"{GIRL_END.get(self.current_target.name, '故事结局未知')}")
                    print("感谢游玩！")
                    break
            else:
                print("无效输入，请重新选择。")

            if self.current_target == self.characters["None"]:
                break


if __name__ == "__main__":
    game = Game()
    game.start()
