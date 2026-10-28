import sys
import os
from character import Character
from save_and_load import save,load

class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target = self.characters["None"]

    def save_game(self):
        save(
            current_target=self.current_target.name,
            affinity=self.current_target.affinity,
            talk_turn=self.current_target.talk_turn,
            give_gift_turn=self.current_target.give_gift_turn
        )

    def load_game(self):
        data = load()
        if data:
            current_target = data['current_target']
            self.current_target = self.characters[current_target]
            self.current_target.affinity = data['affinity']
            self.current_target.talk_turn = data['talk_turn']
            self.current_target.give_gift_turn = data['give_gift_turn']
            print(f'当前路线：【{current_target}线】')
            return True
        return False


    def start(self):
        print("========== 游戏开始：校园 if·恋 ==========")
        if os.path.exists('save.json'):
            choice = input('是否读取存档？（1.是 / 2.否）：')
            if choice == '1':
                if self.load_game():
                    self.story_loop()
                    return

        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩 galgame")

    def scene_senpai(self):
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
            print("\n在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 转身离开\n请选择：")
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        if choice == "1":
            print("\n你上前耐心地帮她剖析了算法逻辑，并给出了完美解法。")
            print("小白崇拜地看着你并向你道谢。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("\n留下小白一个人，你转身离去。")
            return False


    def scene_jiejie(self):
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        # TODO 两种选择 如果选择了 1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回 False
        if choice == "1":
            print("\n你微微一笑，自信而流畅地向她阐述了架构设计。")
            print("姐姐眼中掠过一丝惊艳与欣赏。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("\n你头也不抬地继续敲代码，姐姐见状无奈地轻笑一声，走开了。")
            return False
        
    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 存档")
            print("4. 读档")
            print("5. 离开（退出游戏）")

            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作
            match choice:
                case '1':
                    self.current_target.talk()
                    # 即时存档
                    self.save_game()

                case '2':
                    self.current_target.give_gift()
                    # 即时存档
                    self.save_game()

                case '3':
                    self.save_game()
                    print('\n存档成功')

                case '4':
                    self.load_game()

                case '5':
                    print("你选择离开，游戏结束。")
                    sys.exit(0)

                case _:
                    print("无效输入，请重新选择。")

            if self.current_target == self.characters["None"] or self.current_target.check_ending():
                break


if __name__ == "__main__":
    game = Game()
    game.start()
