import os
import tkinter as tk
from PIL import Image,ImageTk

class Pages:
    """初始化、页面"""

    example = None  # 保存实例作为类变量，方便引用

    def __init__(self,root):

        Pages.example = self

        self.root = root
        self.root.title("校园·if 恋")
        self.root.resizable(False, False)
        self.root.geometry("480x320")

        # 创建容器（储存不同页面）
        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self._file_read_data()

        # 创建存档界面与游戏主界面
        self._create_archives_page()  
        self._create_home_page()


    def _file_read_data(self):
        '''读取数据文件'''
        print("读取文件！")
        import ast
    
        # 确保Data目录存在
        if not os.path.exists('./Data'):
            os.makedirs('./Data')

        # 尝试打开并读取数据文件
        try:
            with open('./Data/GameData.txt','r',encoding='utf-8')as file:  
                game_data = file.read()   
                game_data = ast.literal_eval(game_data)
        except:
            print("❌ 数据文件丢失！")
            game_data = {"archives":[{},{},{},{}],"collections":{"ending":[""]}}  # 初始化数据
            
            try:  
                with open('./Data/GameData.txt','w',encoding='utf-8')as file:
                    file.write(str(game_data))
                    print("✅ 新建数据文件成功！")
            except:
                print("❌ 新建数据文件失败！")

        finally:
            # 用实例变量储存数据，方便读取
            self.archives = game_data["archives"]
            self.collections = game_data["collections"]


    def _file_save_data(self):
        '''写入数据文件'''

        with open('./Data/GameData.txt','w',encoding='utf-8')as file:  
            game_data = {"archives":self.archives,"collections":self.collections}  # 获取临时储存
            file.write(str(game_data))



    """开始页面"""
    def _create_home_page(self):
        self.home_page = tk.Frame(self.container)
        self.home_page.grid(row=0, column=0, sticky="nsew")

        self.home_title = tk.Label(
            self.home_page,
            text="❤ 校园·if 恋 ❤",
            font=("YouYuan",16,"bold")
        )
        self.home_title.place(relx=0.5,rely=0.25,anchor="center")

        # 创建按钮实例并放置
        self.new_game_btn = Buttons("新的游戏",self.home_page,self._new_game)
        self.continue_game_btn = Buttons("继续游戏",self.home_page,self._continue_game)
        self.exit_game_btn = Buttons("退出游戏",self.home_page,self._exit_game)
        self.new_game_btn.display_button(0.5,0.5)
        self.continue_game_btn.display_button(0.5,0.625)
        self.exit_game_btn.display_button(0.5,0.75)

            
    def _new_game(self):  # 新的游戏
        # 待
        self.enter_game(1)

    def _continue_game(self):  # 继续游戏
        valid_archives = []
        
        from datetime import datetime
        for i in range(4):
            if self.archives[i]:    
                valid_archives.append((datetime.strptime(self.archives[i]['sign']['time'], "%Y-%m-%d %H:%M:%S"),i))
        
        # 查找最近一次存档时间
        if valid_archives:
            self.enter_game(0,max(valid_archives, key=lambda x: x[0])[1])  # 用元组来表示具体时间与在原列表的索引
        else:
            print("\a")
            print("⚠ 没有可载入的存档！")
            
    
    def enter_game(self,is_new,save_id=None):
        self._cleanup()

        self._create_game_page()
        self.game_page.lift()  # 将游戏页面提至最上层
        
        import Game_Progress as gp
        self.game = gp.Game(self.game_page)  # 创建新的游戏实例


        if is_new:
            self.game.opening()

        # 从读取的数据中获得并传入游戏
        else:
            import copy
            targeted_data = copy.deepcopy(self.archives[save_id])  # 深拷贝，防止共享字典对象导致修改存档信息

            存档_rejection = targeted_data.get("player", {}).get("key", {}).get("rejection 1", [])

            progress_data = targeted_data.get("progress", {})
            character_data = targeted_data.get("character", {})
            player_data = targeted_data.get("player", {})
            sign_data = targeted_data.get("sign",{})
            
            before_jump_rejection = self.game.characters["你"].attribute.get("key", {}).get("rejection 1", [])

            self.game.jump(
                progress_data.get("phase"),
                progress_data.get("node", 0),
                progress_data.get("variants", 0),
                progress_data.get("text_index", 0),

                character_data,
                player_data,
                
                sign_data.get("background")[1]

            )

            after_jump_rejection = self.game.characters["你"].attribute.get("key", {}).get("rejection 1", [])
            

    def return_home_page(self):  # 返回主页
        self.home_page.tkraise()
        self._cleanup()


    def _cleanup(self):
        """资源清理"""
        import gc

        # 重置回调
        self.choice_callback = None

        # 重置模块
        try:
            import Game_Operation as go
            go.Others.text_framework = None
            go.Character.text_talk = None
        except:
            print("❌ 没有正常重置模块资源！")


        if hasattr(self, 'game'):
            del self.game

        if hasattr(self, 'game_page') and self.game_page:
            self.game_page.destroy()
            self.game_page = None

        gc.collect()  # 回收

    
    def _exit_game(self):  # 退出游戏
        self.root.destroy()

        import sys
        sys.exit()




    """游戏页面"""
    def _create_game_page(self):
        self.game_page = tk.Frame(self.container)
        self.game_page.grid(row=0, column=0, sticky="nsew")

        # 创建背景标签（存放背景图片）
        self.bg_label = tk.Label(self.game_page)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_label.lower()

        # 创建选项按钮
        self.option_A = Buttons("",self.game_page,self._option_choose_A)
        self.option_B = Buttons("",self.game_page,self._option_choose_B)
        self.option_C = Buttons("",self.game_page,self._option_choose_C)
        self.option_D = Buttons("",self.game_page,self._option_choose_D)

        self.setting = Panel(self.game_page)

        self.choice_callback = None  # 初始化选项回调


    def background_setting(self,style):
        '''背景设置'''

        self.bg_id = style

        if type(style) == tuple and len(style) == 2:  # 图片背景

            image_styles = [["./Images/BackGround_0.png"],[],[],[]]

            try:
                self.bg_path = image_styles[style[0]][style[1]]  # 获取图片路径

                bg_image_or = Image.open(self.bg_path)
                bg_image = bg_image_or.resize((480,320))
                self.bg_image_tk = ImageTk.PhotoImage(bg_image)  # 用实例变量保存裁剪后的转换，防止被回收

                self.bg_label.config(image=self.bg_image_tk, bg='systemButtonFace')

                bg_image_or_for_setting = bg_image.crop((bg_image.width-24,0,bg_image.width,24))  # 裁剪游戏背景右上角作为设置按钮的背景
                self.bg_image_tk_for_setting= ImageTk.PhotoImage(bg_image_or_for_setting)
                self.setting.setting_label.config(image=self.bg_image_tk_for_setting,bg='systemButtonFace',compound="center")

            except:
                print("❌ 背景图片导入失败！")
                self.bg_path = self.bg_image_or = self.bg_image_tk = None
                self.bg_label.config(image='', bg='white')  # 如果导入图片失败，使用默认白色背景

                self.setting.setting_label.config(image='', bg='white')

        elif type(style) == int:  # 纯色背景

            color_styles = ['white','black']

            self.bg_path = color_styles[style]
            self.bg_image_or = self.bg_image_tk = None

            self.bg_label.config(image='', bg=self.bg_path)
            self.setting.setting_label.config(image='', bg=self.bg_path)


    def display_option(self,text: list,callback=None):
        '''显示选项'''

        options = [self.option_A, self.option_B, self.option_C, self.option_D]
        positions = [
            [(0.5,0.5)],
            [(0.5,0.375),(0.5,0.625)],
            [(0.5,0.3),(0.5,0.5),(0.5,0.7)],
            [(0.25,0.375),(0.75,0.375),(0.25,0.625),(0.75,0.625)]
            ]

        for i, option in enumerate(options):
            if i < len(text):
                option.button.config(text=text[i])
                option.display_button(*positions[len(text)-1][i])

        self.choice_callback = callback  # 设置回调函数（下一步操作）


    def _option_choose(self,choice):

        for option in [self.option_A, self.option_B, self.option_C, self.option_D]:
            option.hide_button()

        if self.choice_callback:
            self.choice_callback(choice)

        self.choice_callback = None

    def _option_choose_A(self): self._option_choose('A')
    def _option_choose_B(self): self._option_choose('B')
    def _option_choose_C(self): self._option_choose('C')
    def _option_choose_D(self): self._option_choose('D')
        


    '''存档页面'''
    def _create_archives_page(self):
        self.archives_page = tk.Frame(self.container)
        self.archives_page.grid(row=0, column=0, sticky="nsew")
        self.save_or_load = 0

        self._create_exit_label()
        self._create_archives_frame()


    def _create_exit_label(self):
        '''创建并放置退出标签'''

        self.exit_savepage_label = tk.Label(
            self.archives_page, 
            text="➾",
            font=("Arial", 12),
            fg='#333333',
            cursor='hand2',
            relief='flat',
            bd = 0,
            padx = 0,
            pady = 0
        )
        self.exit_savepage_label.place(relx=1,rely=0,anchor="ne",width=24,height=24)
        self.exit_savepage_label.bind("<Button-1>",self._exit_archives_page)
    
    def _exit_archives_page(self,event=None):
        self.archives_page.lower()
        if self.game_page.winfo_exists():
            self.game_page.lift()


    def _create_archives_frame(self):
        '''创建并初始化存档框架'''

        archives = []
        archive_time_labels = []

        for i in range(4):

            archive = tk.Frame(self.archives_page,  # 框架
                            bg="lightgray",
                            relief='groove',
                            cursor='hand2')
            
            archive.place(relx=0.25 + 0.5 * (i % 2), 
                          rely=0.25 + 0.5 * (i //2),
                          anchor="center", 
                          width=120, height=80)


            
            archive.bind("<Button-1>", getattr(self, f'_click_archive_{i}'))  # 动态属性
            archives.append(archive)

            
            archive_time_label = tk.Label(archives[i],text="",  # 时间、背景组件
                                        fg="gold",
                                        font=("Arial",8),
                                        bd=2,padx=0,pady=0,
                                        compound='center',relief='sunken')
                
            archive_time_label.place(relx=0.5,rely=0.5,x=0, y=0, relwidth=1, relheight=1,anchor="center")
                
            archive_time_label.bind("<Button-1>", getattr(self, f'_click_archive_{i}'))
            archive_time_labels.append(archive_time_label)
                
            sign = self.archives[i].get('sign', {})

            if sign:
                sign = self.archives[i]['sign']  # 获取存档时间与背景
                background = sign['background'][0]

                archive_time_labels[i].config(text = sign['time'])

                setattr(self, f'archive_time_{i}', sign['time'])  # 标记存档时间
                setattr(self, f'archive_bg_{i}', background)  # 标记背景路径

                if background in ["white","black"]:
                        archive_time_label.config(bg = background)

                else:
                    try:
                        setattr(self, f'archive_bg_img_tk_{i}', ImageTk.PhotoImage(Image.open(background).resize((120, 80))))  # 储存背景图片
                        archive_time_labels[i].config(image = getattr(self, f'archive_bg_img_tk_{i}'))
                            
                    except:
                        print("❌ 背景图片获取失败！")
                        setattr(self, f'archive_bg_img_tk_{i}',None)
                        archive_time_labels[i].config(bg = 'white')

    
        self.archive_0, self.archive_1, self.archive_2, self.archive_3 = archives  # 同时赋值给单个属性
        self.archive_time_label_0, self.archive_time_label_1, self.archive_time_label_2, self.archive_time_label_3, = archive_time_labels

        
                    
    '''点击存档'''
    def _click_archive(self,save_id):
        if self.save_or_load == 0:  # 存档

            self._temporary_storage(save_id)

            try:

                if "/" in self.bg_path :
                    setattr(self, f'archive_bg_img_tk_{save_id}', ImageTk.PhotoImage(Image.open(self.bg_path).resize((120, 80))))  # 使用当前背景图片
                    
                    getattr(self, f'archive_time_label_{save_id}').config(
                        anchor='center',
                        image=getattr(self, f'archive_bg_img_tk_{save_id}'),  # 显示存档背景
                        text=getattr(self, f'archive_time_{save_id}'))  # 显示存档时间
                    
                else:
                    setattr(self, f'archive_bg_img_tk_{save_id}', None)

                    getattr(self, f'archive_time_label_{save_id}').config(

                        bg=self.bg_path, text=getattr(self, f'archive_time_{save_id}'))
                    
            except:
                print("❌ 背景图片获取失败！")

                setattr(self, f'archive_bg_img_tk_{save_id}', None)
                
                getattr(self, f'archive_time_label_{save_id}').config(

                        bg='white', text=getattr(self, f'archive_time_{save_id}'))
            
        else:  # 读档
            if self.archives[save_id]:
                self.enter_game(0, save_id)
            else:
                print("\a")
                print("⚠ 此处未存档，不可载入！")


    def _click_archive_0(self,event=None): self._click_archive(0)
    def _click_archive_1(self,event=None): self._click_archive(1)
    def _click_archive_2(self,event=None): self._click_archive(2)
    def _click_archive_3(self,event=None): self._click_archive(3)


        
    def _temporary_storage(self,save_id=0):
        '''临时、局部存储'''


        # 获取类型以判断索引
        import Game_Operation as go

        if self.game.text_type == 0:
            text_index = go.Others.text_framework.text_index
        elif self.game.text_type == 1:
            text_index = go.Character.text_talk.text_index
        else:
            text_index = 0


        import datetime
        save_time = str(datetime.datetime.now()).split('.')[0]  # 获取保存时间
        setattr(self,f'archive_time_{save_id}',save_time) 

        setattr(self,f'archive_bg_{save_id}',self.bg_path)  # 储存图片路径
        
        
        # 提取游戏信息
        progress = {"phase":self.game.current_phase,"node":self.game.current_node,"variants":self.game.current_variants,"text_index":text_index}
        character = {"学姐": self.game.characters["学姐"].attribute ,
                     "小白": self.game.characters["小白"].attribute ,
                     "姐姐": self.game.characters["姐姐"].attribute}
        player = self.game.characters["你"].attribute
        sign = {"time": save_time ,"background": (self.bg_path,self.bg_id)}


        self.archives[save_id] = {
                            "progress":progress,
                            "character":character,
                            "player":player,
                            "sign":sign
                        }
        current_rejection = self.game.characters["你"].attribute.get("key", {}).get("rejection 1", [])

        self._file_save_data()  # 紧跟全局、文件形式储存



class Panel:
    """面板"""

    '''设置面板'''
    def __init__(self,root):
        self.root = root
        self.command = self.switch

        self._create_setting_label()
        self._create_setting_frame()


    def _create_setting_label(self):
        '''创建并放置设置标签'''
        self.setting_label = tk.Label(
            self.root, 
            text="⚙️",
            font=("Arial", 12),
            fg='#333333',
            cursor='hand2',
            relief='flat',
            bd = 0,
            padx = 0,
            pady = 0
        )

        self.setting_label.place(relx=1,rely=0,anchor="ne",width=24,height=24)
        self.setting_label.bind("<Button-1>",self.switch)

    def _create_setting_frame(self):
        '''创建设置面板'''
        self.setting_frame = tk.Frame(self.root,bg='lightgray',relief='groove',bd=1,width=120, height=40)
    
        # 创建面板按钮
        self.save_progress = Buttons('存档',self.setting_frame,self.save_progress_operate,1)
        self.load_progress = Buttons('读档',self.setting_frame,self.load_progress_operate,1)
        self.return_main_page = Buttons('返回',self.setting_frame,self.suspend_progress_operate,1)

        self.save_progress.display_button(0.19,0.5)
        self.load_progress.display_button(0.5,0.5)
        self.return_main_page.display_button(0.81,0.5)

    def switch(self,event=None):
        '''点击开关'''
        if self.setting_frame.winfo_viewable():
            self.setting_frame.place_forget()
        else:
            self.setting_frame.place(relx=0.78, rely=0, anchor='n')

    
    '''按钮具体操作'''
    def save_progress_operate(self):
        Pages.example.archives_page.tkraise()
        Pages.example.save_or_load = 0

    def load_progress_operate(self):
        Pages.example.archives_page.tkraise()
        Pages.example.save_or_load = 1

    def suspend_progress_operate(self):
        Pages.example.return_home_page()



class Buttons:
    """按钮"""

    def __init__(self,text,root,command,buttons_type=0):
        self.text = text
        self.root = root
        self.command = command
        self.buttons_type = buttons_type
        self._create_button()


    def _create_button(self):
        self.button = tk.Button(self.root,
                                text=self.text,
                                command=self.command,
                                cursor='hand2',
                                **BUTTONS_STYLE[self.buttons_type])
    
    def display_button(self,x,y):
        self.button.place(relx=x,rely=y,anchor="center")
    
    def hide_button(self):
        self.button.place_forget()



'''按钮样式'''
BUTTONS_STYLE = ({
    'font': ("YouYuan", 12),
    'fg': '#000000',
    'bg': '#FFFFFF',
    'padx': 12,
    'pady': 4,
    'relief': 'raised',
    'overrelief': 'ridge',
    'bd': 2
    },{
    'font': ("YouYuan", 8),
    'fg': "#000000",
    'bg': 'lightgray',
    'padx': 4,
    'pady': 4,
    'relief': 'flat',
    'overrelief': 'sunken',
    'bd': 1
    })