import os
import random
import sys
import Dialogue
from Characters import Character
import save


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target = None
        save.current_data = {'character': None, 'affinity': 0, 'story_progress': '1.1', 'choice':{}}



    def start(self):
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        is_load_game = input('是否加载上次存档？(y/n): ')
        if is_load_game == 'y':
            try:
                save.current_data = save.loadgame()
                print(save.current_data)
                self.current_target = self.characters[save.current_data["character"]]
                self.current_target.affinity = save.current_data['affinity']
            except Exception as e:
                print(f"加载存档失败: {e}")
            self.story_loop()
        else:
            pass

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self):
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            #增加5点初遇好感度
            self.current_target.change_affinity(5)
            save.current_data['join_club'] = False
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
        if choice == "1":

            print("\n你一把拉过她的笔记本，几行代码迅速解决了问题。(最终幻想)")
            print("小白虎躯一震。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            save.current_data['character'] = '小白'
            #增加5点初遇好感度
            self.current_target.change_affinity(5)
            self.story_loop()
            return True
        else:
            print("你脑中心想:这也不会？算了，算了。")
            return False


    def scene_jiejie(self):
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你淡淡一笑，娓娓道来，姐姐听得入神。")
            print("姐姐眼神一亮。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            save.current_data['character'] = '姐姐'
            #增加5点初遇好感度
            self.current_target.change_affinity(5)
            self.story_loop()
            return True
        else:
            print("你是坚定的西格玛男人，没有理会这个奇怪的女人。")
            return False
        
    def story_loop(self):
        """角色线主循环"""
        dialogue = Dialogue.Dialogue(self.current_target,save.current_data['story_progress'])

        while True:
            save.quicksave()
            self.current_target.check_ending()
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")
            print("5. 读取存档")
            print("6. 保存存档")

            choice = input("请输入选项：")

            if choice == "0":
                self.current_target.change_affinity(int(input('添加好感:')))
                continue

            #输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                #尝试触发主线
                if not dialogue.start_dialogue(save.current_data['story_progress']):
                    #主线不满足，随机支线
                    daily_dialogue_list = ['100','101','102']
                    rand_dialogue = random.choice(daily_dialogue_list)
                    while not dialogue.start_dialogue(rand_dialogue):
                        daily_dialogue_list.remove(rand_dialogue)
                        rand_dialogue = random.choice(daily_dialogue_list)
                        if len(daily_dialogue_list) == 0:
                            print("当前没有可触发的对话。")
                            break
                continue


            #输入2----
            if choice == "2":
                print('你想送给她什么礼物？')
                for i in Character.GIFT_EFFECTS.keys():
                    print(i)
                gift = input("请输入礼物名称：")
                self.current_target.give_gift(gift)
                continue


            #输入3----
            if choice == "3":
                print(f"{self.current_target.name}当前好感度：{self.current_target.affinity}")
                continue



            if choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            if choice == "5":
                filename = input("存档名:")
                try:
                    data = save.loadgame(os.path.join(os.getcwd(),'save',filename+'.json'))
                    print(data)
                    self.current_target = self.characters[data["character"]]
                    self.current_target.affinity = data['affinity']
                except Exception as e:
                    print(f"加载存档失败: {e}")
                continue

            if choice == "6":
                filename = input("存档名:")

                try:
                    save.current_data['character'] = self.current_target.name
                    save.current_data['affinity'] = self.current_target.affinity
                    save.savegame(save.current_data,os.path.join(os.getcwd(),'save',filename+'.json'))
                    print("存档成功")
                except Exception as e:
                    print(f"存档失败: {e}")
                continue



            else:
                print("无效输入，请重新选择。")

            










if __name__ == "__main__": 
    game = Game()
    game.start()
