from typing import Dict,Optional
from manage import Character
import sys
from story import DIALOGUES,GIFT_EFFECTS
import copy

class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target = None
        self.saves ={}

    def start(self):
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
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice =='1':
            print('\n你走上前去，主动在她身边说出一个思路，小白顿时恍然大悟')
            print('小白看你的眼神顿时发着光，主动要加你的企鹅。你进入【小白线】')
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("你：『哎，这种题目你只要这样这样就做出来啦』")
            print('小白感觉尊严受到了屈辱，潸然泪下，哭着骂你，你知道自己说错了话，匆忙逃走了')
            return False
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        #注意 除了判断外 你可以同时输出角色的反应 
        #比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        #写法可以借鉴学姐线

    def scene_jiejie(self):
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice =='1':
            print('你：『唉只是些小想法罢了，只要这样这样』说了一下大概思路')
            print('姐姐看着你的做法思路非常赞赏，又问了一些代码上的问题也被你一一解答，顿时两眼放光')
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你头也不抬，保持敲代码的状态")
            print("姐姐深感无趣，离开了咖啡店")
            return False
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        #要求同上
        
    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作 

            #输入1---关于聊天的内容可以自己构思 也可以从剧本中截取
            if choice == "1":
                self.current_target.talk()
            #输入2----
            elif choice == '2':
                self.current_target.give_gift()
            #输入3----
            elif choice == '3':
                print(f"{self.current_target.name} 当前好感度：{self.current_target.affinity}")
            elif choice == '4':
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                break
# 存档读档功能的添加
def play (initial_game: Optional[Game] = None):
    game = initial_game if initial_game is not None else Game()
    #初始存档
    incoming = yield game
    if incoming:
        game = incoming
    print("========== 游戏开始：校园 if·恋 ==========")
    print("你是一名刚刚踏入大学校园的新生。")
    print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
    print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
    print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
    print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")
    incoming = yield game
    if incoming:
        game = incoming
    print("\n【场景一：社团学姐】")
    print("你路过社团活动室，学姐正拿着画板注意到你。")
    print("学姐：『这位新生？要不要来试试？』")
    incoming = yield game
    if incoming:
        game = incoming
    choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
    if choice == "1":
        print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
        game.current_target = game.characters["学姐"]
    else:
        print("在纵目睽睽下，你扬长而去。")
        game.current_target = None
    incoming = yield game
    if incoming:
        game = incoming

    if game.current_target is None:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")
        incoming = yield game
        if incoming:
            game = incoming
        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == '1':
            print('\n你走上前去，主动在她身边说出一个思路，小白顿时恍然大悟')
            print('小白看你的眼神顿时发着光，主动要加你的企鹅。你进入【小白线】')
            game.current_target = game.characters["小白"]
        else:
            print("你：『哎，这种题目你只要这样这样就做出来啦』")
            print('小白感觉尊严受到了屈辱，潸然泪下，哭着骂你，你知道自己说错了话，匆忙逃走了')
            game.current_target = None
    incoming = yield game
    if incoming:
        game = incoming

    if game.current_target is None:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")
        incoming = yield game
        if incoming:
            game = incoming
        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == '1':
            print('你：『唉只是些小想法罢了，只要这样这样』说了一下大概思路')
            print('姐姐看着你的做法思路非常赞赏，又问了一些代码上的问题也被你一一解答，顿时两眼放光')
            game.current_target = game.characters["姐姐"]
        else:
            print("你头也不抬，保持敲代码的状态")
            print("姐姐深感无趣，离开了咖啡店")
            game.current_target = None

    if game.current_target is None:
        print("\n啥，眼前三妹子都不要？？死现充别玩galgame")
        print("游戏到此结束。")
        return game
    while True:
        incoming = yield game
        if incoming:
            game = incoming
        target: Character = game.current_target
        print(f"\n你要做什么？（当前目标：{target.name}）")
        print("1. 和她聊天")
        print("2. 送她礼物")
        print("3. 查看好感度")
        print("4. 离开（退出游戏）")
        incoming = yield game
        if incoming:
            game = incoming
        choice = input("请输入选项：")

        if choice == "1":
            target.talk()
        elif choice == "2":
            target.give_gift()
        elif choice == "3":
            print(f"{target.name} 当前好感度：{target.affinity}")
        elif choice == "4":
            print("你选择离开，游戏结束。")
            sys.exit(0)
        else:
            print("无效输入，请重新选择。")

        incoming = yield game
        if incoming:
            game = incoming

        if target.check_ending():
            print("进入结局线，当前角色线结束。")
            break

    print("角色线结束，游戏结束。")
    incoming = yield game
    if incoming:
        game = incoming
    return game
#交互式运行器
def interactive_run():
    g = play()
    try:
        state = next(g)
    except StopIteration:
        print("生成器结束")
        return
    saves = {}
    try:
        while True:
            print("操作:\na.继续\ns(n).存档\nl(n).读档")
            cmd = input('>').strip()
            if cmd == 'a':
                state = g.send(None)
            elif cmd[0]=='s' and len(cmd)>=4:
                slot = int(cmd[2:(len(cmd)-1)])
                saves[slot] = copy.deepcopy(state)
                print(f"已经存档在槽{slot}")
                
            elif cmd[0]=='l' and len(cmd)>=4:
                slot = int(cmd[2:(len(cmd)-1)])
                if slot not in saves:
                    print(f"此处无存档")
                    continue
                loaded = copy.deepcopy(saves[slot])
                state = g.send(loaded)
            else:
                print("错误")
    except StopIteration:
        print("结束")

    
if __name__ == "__main__":
    interactive_run()
