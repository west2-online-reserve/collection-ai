import Game_Operation as go
import Text_Framework as tf
from System_Control import Pages



class Game:
    """游戏进程"""

    def __init__(self,root):

        self.characters = {
            "学姐": go.Character("学姐", "社团里的艺术少女",root),
            "小白": go.Character("小白", "课堂上的元气同学",root),
            "姐姐": go.Character("姐姐", "食堂里的温柔姐姐",root),
            "你": go.Player("你", "神秘的新生代表？",root)
        }
        self.root = root


        # 当前场景、节点、分支
        self.current_phase = None
        self.current_node = 0
        self.current_variants = 0  # 当前索引是Text_Framework里面的一个实例变量

        # 起始场景、节点、分支、索引
        Game.start_phase = None
        Game.start_node = 0
        Game.start_variants = 0
        Game.start_index = 0

        Game.is_first_operation = True


    def execute(self,node,*next_step,variants=0):
        '''统一推进剧情操作与节点记录'''
        self.current_node = node
        self.current_variants = variants
        
        if Game.is_first_operation:
            index = Game.start_index
            Game.is_first_operation = False
        else:
            index = 0

        current_phase_operations = go.SCRIPT.get(self.current_phase)
        current_operation = current_phase_operations[node]
        self.text_type = ["narration","talk","choose"].index(current_operation.get("type"))
        if self.text_type == 0:
            go.Others.narration(self.root,[self.current_phase,node],next_step[0],variants,index)
        if self.text_type == 1:
            self.characters[current_operation.get("character")].talk([self.current_phase,node],next_step[0],variants,index)
        if self.text_type == 2:
            go.Others.choose([self.current_phase,node],*next_step,variants=variants)  # 解包元组再次传入


    def jump(self,phase,node,variants,index,character,player,background):
        '''加载读档信息，跳转画面'''

        Game.is_first_operation = True

        Game.start_phase = phase
        Game.start_node = node
        Game.start_variants = variants
        Game.start_index = index
        
        self.characters['学姐'].attribute = character['学姐']
        self.characters['小白'].attribute = character['小白']
        self.characters['姐姐'].attribute = character['姐姐']
        self.characters['你'].attribute = player


        func = {"phase 0":self.opening,"phase 1r1":self.scene_senpai,"phase 1r2":self.scene_xiaobai,"phase 1r3":self.scene_jiejie,
                "phase 0a1":self.choose_again,"phase 0a2":self.choose_again_again}.get(phase,self.opening)
        
        from System_Control import Pages
        Pages.example.background_setting(background) 

        func()



    def opening(self):
        self.current_phase = "phase 0"

        p0 = [lambda: (Pages.example.background_setting(0),self.execute(0,p0[1])),
              lambda : (Pages.example.background_setting((0,0)),self.execute(1,p0[2])),
              lambda : self.execute(2,p0[3]),
              lambda : self.execute(3,p0[4][0],p0[4][1],p0[4][2]),
              [lambda : self.execute(4,self.scene_senpai,variants=0),
              lambda : self.execute(4,self.scene_xiaobai,variants=1),
              lambda : self.execute(4,self.scene_jiejie,variants=2)]]
        
        if Game.is_first_operation:
            if Game.start_node == 4:
                p0[Game.start_node][Game.start_variants]()
            else:
                p0[Game.start_node]()
        else:
            p0[0]()


        
        # 选择进入三个场景 如果三个都不选....
        '''if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")'''


    def scene_senpai(self):
        self.current_phase = "phase 1r1"

        def bad_options():
            self.characters["你"].attribute["key"]["rejection 1"].append("学姐")
            r:list = self.characters["你"].attribute["key"]["rejection 1"]
            if len(r) == 1:
                self.choose_again(variants = 0)
            elif len(r) == 2:
                if "小白" in r:
                    next_variants = 2
                if "姐姐" in r:
                    next_variants = 1
                self.choose_again_again(variants = next_variants)    
            else:
                self.bad_ending_0()  # 全部拒绝则进入坏结局

        p1r1 = [lambda: (Pages.example.background_setting(0),self.execute(0, p1r1[1])),
            lambda: self.execute(1, p1r1[2]),
            lambda: self.execute(2, p1r1[7], p1r1[3]),
            lambda: self.execute(3, p1r1[4]),
            lambda: self.execute(4, p1r1[5]),
            lambda: self.execute(5, p1r1[7], p1r1[6]),
            lambda: self.execute(6, bad_options),
            lambda: self.execute(7, lambda: None)]
    
        if Game.is_first_operation:
            p1r1[Game.start_node]()
        else:
            p1r1[0]()

    def scene_xiaobai(self):
        self.current_phase = "phase 1r2"

        def bad_options():
            self.characters["你"].attribute["key"]["rejection 1"].append("小白")
            r:list = self.characters["你"].attribute["key"]["rejection 1"]
            if len(r) == 1:
                self.choose_again(variants = 1)
            elif len(r) == 2:
                if "学姐" in r:
                    next_variants = 2
                if "姐姐" in r:
                    next_variants = 0
                self.choose_again_again(variants = next_variants)    
            else:
                self.bad_ending_0()  # 全部拒绝则进入坏结局

        p1r2 = [lambda: (Pages.example.background_setting(0),self.execute(0, p1r2[1])),
                lambda: self.execute(1, p1r2[2]),
                lambda: self.execute(2, p1r2[7], p1r2[3]),
                lambda: self.execute(3, p1r2[4]),
                lambda: self.execute(4, p1r2[5]),
                lambda: self.execute(5, p1r2[7], p1r2[6]),
                lambda: self.execute(6, bad_options),
                lambda: self.execute(7, lambda: None)]
        
        if Game.is_first_operation:
            p1r2[Game.start_node]()
        else:
            p1r2[0]()

    def scene_jiejie(self):
        self.current_phase = "phase 1r3"

        def bad_options():
            self.characters["你"].attribute["key"]["rejection 1"].append("姐姐")
            r:list = self.characters["你"].attribute["key"]["rejection 1"]
            if len(r) == 1:
                self.choose_again(variants = 2)
            elif len(r) == 2:
                if "学姐" in r:
                    next_variants = 1
                if "小白" in r:
                    next_variants = 0
                self.choose_again_again(variants = next_variants)    
            else:
                self.bad_ending_0()  # 全部拒绝则进入坏结局

        p1r3 = [lambda: (Pages.example.background_setting(0),self.execute(0, p1r3[1])),
            lambda: self.execute(1, p1r3[2]),
            lambda: self.execute(2, p1r3[7], p1r3[3]),
            lambda: self.execute(3, p1r3[4]),
            lambda: self.execute(4, p1r3[5]),
            lambda: self.execute(5, p1r3[7], p1r3[6]),
            lambda: self.execute(6, bad_options),
            lambda: self.execute(7, lambda: None)]
    
        if Game.is_first_operation:
            p1r3[Game.start_node]()
        else:
            p1r3[0]()


    def choose_again(self, variants=None):
        if Game.is_first_operation: 
            variants = Game.start_variants  # 初始不传递variants，赋值Game.start_variants
        
        self.current_phase = "phase 0a1"
        Pages.example.background_setting(0)

        p0a1 = [lambda: self.execute(0, p0a1[1], variants=variants),
                lambda: self.execute(1, p0a1[2], variants=variants),
                lambda: self.execute(2, p0a1[3][0], p0a1[3][1], variants=variants),
                [lambda: self.execute(3, self.scene_senpai, variants=0),
                lambda: self.execute(3, self.scene_xiaobai, variants=1),
                lambda: self.execute(3, self.scene_jiejie, variants=2)]]
        
        # 根据variants删除已去过的选项
        available_options = p0a1[3][:]  # 复制列表
        del available_options[variants]
        p0a1[3] = available_options
        
        if Game.is_first_operation:
            if Game.start_node == 3:
                p0a1[Game.start_node][Game.start_variants]()
            else:
                p0a1[Game.start_node]()
        else:
            p0a1[0]()

    def choose_again_again(self, variants=None):
        if Game.is_first_operation: 
            variants = Game.start_variants
        
        self.current_phase = "phase 0a2"
        Pages.example.background_setting(0)

        p0a2 = [lambda: self.execute(0, p0a2[1], variants=variants),
                lambda: self.execute(1, p0a2[2], variants=variants),
                lambda: self.execute(2, p0a2[3][0], variants=variants),
                [lambda: self.execute(3, self.scene_senpai, variants=0),
                lambda: self.execute(3, self.scene_xiaobai, variants=1),
                lambda: self.execute(3, self.scene_jiejie, variants=2)]]
        
        # 只保留唯一的选项
        unselected_option = p0a2[3][variants]
        p0a2[3] = [unselected_option]
        
        if Game.is_first_operation:
            if Game.start_node == 3:
                p0a2[Game.start_node][Game.start_variants]()
            else:
                p0a2[Game.start_node]()
        else:
            p0a2[0]()


    def bad_ending_0(self):

        self.current_phase = "bad_ending 0"
        Pages.example.background_setting(0)
        Pages.example.setting.setting_label.place_forget()
        
        pbe0 = [lambda: self.execute(0, pbe0[1]),
                lambda: self.execute(1, pbe0[2]),
                lambda: self.execute(2, lambda: Pages.example.return_home_page())]
        
        pbe0[0]()





    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")

            choice = input("请输入选项：")

            # TODO 完成输入不同选项时 进行的操作 

            #输入1---关于聊天的内容可以自己构思 也可以从剧本中截取



            #输入2----



            #输入3----
            if True:
                pass


            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                break