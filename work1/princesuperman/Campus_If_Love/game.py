import sys
from typing import Dict,Optional
from character import Character
from save_manager import Savemanager

class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_character:Optional[Character]=None
        self.save_manager=Savemanager()
        #对话内容
        self.dialogues= {
            "学姐": [
        {"text": "你是新来的吧？画画过吗？", "optionA": "其实我更擅长写代码，但我愿意试试画画。", "optionB": "画画？那种小孩子的东西我可没兴趣。"},
        {"text": "这幅画……构图还算不错，你是随便画的吗？", "optionA": "嗯，我是随手画的，没想到能得到你的认可。", "optionB": "不，我可是很认真的！因为你在看嘛。"},
        {"text": "加入社团后，不能三天打鱼两天晒网。", "optionA": "放心吧，我会认真对待。", "optionB": "社团只是玩玩而已吧？不要太严肃了。"},
        {"text": "我平时比较严格，你会不会觉得我很不好相处？", "optionA": "严格才好啊，我喜欢有目标的人。", "optionB": "嗯……确实有点难相处。"},
        {"text": "今天的社团活动结束了，你要不要和我一起收拾？", "optionA": "当然可以，我来帮你。", "optionB": "算了吧，我还有事要做。"}
            ],
            "小白": [
        {"text": "Python 里的 for 循环，我总是写不对……", "optionA": "来，我教你一个小技巧！", "optionB": "这种简单的东西，你都不会？"},
        {"text": "你看我借了好多书，会不会显得很傻？", "optionA": "不会啊，这说明你很爱学习，很可爱。", "optionB": "嗯……的确有点太贪心了。"},
        {"text": "写代码的时候，我总喜欢喝奶茶……你呢？", "optionA": "我也是！来，下次我请你一杯。", "optionB": "我只喝水，健康第一。"},
        {"text": "你会不会觉得我太依赖你了？", "optionA": "依赖我也没关系，我喜欢这种感觉。", "optionB": "嗯……是有点吧。"},
        {"text": "要不要一起留在图书馆自习？我想有人陪。", "optionA": "好啊，我正好也要复习。", "optionB": "算了，我还是回宿舍打游戏吧。"}
            ],
            "姐姐": [
        {"text": "你也喜欢在咖啡店看书吗？", "optionA": "是啊，这里很安静，很适合思考。", "optionB": "其实我只是随便找个地方坐。"},
        {"text": "研究生的生活，其实没有你想象的那么光鲜。", "optionA": "我能理解，压力一定很大吧。", "optionB": "那我还是不要考研了。"},
        {"text": "你觉得，什么样的人最值得依靠？", "optionA": "稳重冷静的人。", "optionB": "长得好看的人。"},
        {"text": "我常常一个人坐到很晚，你不会觉得孤单吗？", "optionA": "有时候孤单是种享受。", "optionB": "一个人太寂寞了，我受不了。"},
        {"text": "如果有一天，我遇到困难，你会帮我吗？", "optionA": "当然会，你不用一个人扛着。", "optionB": "我大概帮不了你吧。"}
            ]
        }
    
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
            self.current_character = self.characters["学姐"]
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
            print("耐心讲解解题的思路，小白恍然大悟。")
            print("小白表现出崇拜的目光。你进入【小白线】！")          
            self.current_character = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("假装没看见")
            return False
    
    def scene_jiejie(self):
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：") 
        if choice == "1":
            print("解释代码的逻辑")
            print("姐姐认真听着。你进入【姐姐线】！")
            self.current_character = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("没有理会")
            return False
    
    #主页
    def show_menu(self)->None:
        print("1.开始游戏")
        print("2.继续游戏")
        print("3.读档")
        print("4.退出游戏")
        choice=input("请选择：")
        if choice=="1":
            self.start_new_game()
        elif choice=="2":
            self.continue_game()
        elif choice=="3":
            self.quick_load_game()
        else:
            print("无效的选择")
            self.show_menu()
    
    #开始新的游戏
    def start_new_game(self)->None:
        self.__init__()  
        self.start() 
    
    #继续游戏
    def continue_game(self)->None:
        saved_data=self.save_manager.load_game()
        if saved_data:
            self.load_game_state(saved_data)
            print("继续上次游戏")
            if self.current_character:
                print(f"角色：{self.current_character.name}")
                print(f"好感度{self.current_character.affinity}")
                self.story_loop()
            else:
                self.start()
        else:
            print("没有存档")
            self.start_new_game()

    #读档
    def quick_load_game(self)->None:
        saved_data=self.save_manager.quick_load()
        if saved_data:
            self.load_game_state(saved_data)
            print("读档成功")
            if self.load_game_state(saved_data):
                print(f"角色：{self.current_character.name}")
                print(f"好感度：{self.current_character.affinity}")
                self.story_loop()
            else:
                self.start()
        else:
            print("无存档")
            self.show_menu()

    #查看好感度
    def show_affinity_info(self)->None:
        if self.current_character:
            print(f"==={self.current_character.name}的好感度信息===")
            print(F"好感度：{self.current_character.affinity}/100") 
       
    #角色线主循环
    def story_loop(self)->None:
        while True:
            print(f"角色：{self.current_character.name}")
            print(f"好感度：{self.current_character.affinity}")
            print("\n你要做什么？")
            print("1.对话")
            print("2.送礼")
            print("3.查看好感度")
            print("4.存档")
            print("5.返回主页")

            choice=input("请选择：")
            if choice=="1":
                self.current_character.talk(self.dialogues[self.current_character.name])
                self.save_manager.auto_save(self.get_game_state())
            elif choice=="2":
                print("可选礼物：鲜花，编程笔记，奶茶，奇怪的石头，精致的钢笔，可爱玩偶，夜宵外卖")
                gift=input("输入礼物名称：")
                self.current_character.give_gift(gift)
                self.save_manager.auto_save(self.get_game_state())
            elif choice=="3":
                self.show_affinity_info()
            elif choice=="4":
                if self.save_manager.quick_save(self.get_game_state()):
                    print("存档成功")
            elif choice=="5":
                self.show_menu()
                break
            else:
                print("重新选择")
            if self.current_character.check_ending():
                self.show_ending()
                break
    # 游戏状态
    def get_game_state(self,game_state:Dict)->Dict:
        for name,char_state in game_state["characters"].items():
            if name in self.characters:
                self.current_character=self.characters[current_name]
        current_name=game_state["current_character"]
        if current_name:
            self.current_character=self.characters[current_name]
    # 加载
    def load_game_state(self,game_state:Dict)->None:
        for name,char_state in game_state["characters"].items():
            if name in self.characters:
                self.characters[name].load_status(char_state)
        current_name=game_state["current_character"]
        if current_name:
            self.current_character=self.characters[current_name]

    # 结局
    def show_ending(self)->None:
        endings={"学姐":"共同进步","小白":"走向未来","姐姐":"一起读研"}
        print(endings[self.current_character.name])
        print(f"好感度：{self.current_character.affinity}")
        input("回车键返回")
        self.show_menu()   

