import tkinter as tk
import time



class TextFramework:
    '''文本管理'''
    def __init__(self,root,text_type:int = 1,name=""):
        '''实例初始化'''
        self.root = root

        # 确定文字种类与样式
        self.text_type = text_type 
        self.style = TEXT_STYLE[text_type]
        self.location = TEXT_LOCATION[text_type]

        self.text_finished = None  # 初始化回调函数（下一步操作）
        
        if text_type == 1:
            self.name = name
            self._create_text_box()

        # 创建并放置文本标签
        self.text_label = tk.Label(self.text_box if text_type == 1 else self.root, 
                                   text="", font=("Segoe UI" if text_type == 1 else "YouYuan", self.style[0]),
                                   wraplength=self.style[1],justify=self.style[2],
                                   fg=self.style[3],bg=self.style[4],relief='raised' if self.text_type==2 else 'flat')
        self.text_label.place(relx=self.location[0], rely=self.location[1], anchor=self.location[2])

        if text_type == 1:
            # 绑定鼠标事件
            self.text_box.bind("<Button-1>", self._on_click)
            self.text_label.bind("<Button-1>", self._on_click)
            self.continue_label.bind("<Button-1>", self._on_click)

            self.last_click_time = 0  # 设置上一次点击时间


    def animation_text(self,texts:list,text_index=0):
        '''文字动画'''
        self.root.update_idletasks()  # 清理事件队列
        self.task_id = None

        # 动态文本索引
        self.text_index = text_index
        self.char_index = 0

        self.is_typing = True

        self.texts = texts
        self.current_text = self.texts[self.text_index]

        if self.text_type == 1:
            self.display_text_box()
        self._type_next_char()

        
    def _type_next_char(self):
        '''打字动画'''
        if self.char_index < len(self.current_text):
            displayed_text = self.current_text[:self.char_index + 1]
            self.text_label.config(text=displayed_text)
            self.char_index += 1
            self.task_id = self.root.after(80 if self.text_type== 1 else 240,self._type_next_char)  # 标记任务
        else:
            self.is_typing = False
            self.task_id = None
            if self.text_type == 0:
                self.root.after(1600, self._to_next_text)
            else:
                self.continue_label.place(relx=1,rely=1,anchor='se')
                self._animation_continue_blink()

    def _to_next_text(self):
        '''文本切换'''
        self.char_index = 0
        self.text_index += 1
        if self.text_index < len(self.texts):
            self.current_text = self.texts[self.text_index]
            self.is_typing = True
            if self.text_type == 1:
                self.continue_label.place_forget()
            self._type_next_char()
        else:
            self.text_label.place_forget()
            if self.text_type == 1:
                self.hid_text_box()
            if self.text_finished:
                self.text_finished()  # 执行回调函数（下一步操作）


    def _create_text_box(self):
        # 创建文本框
        self.text_box = tk.Frame(self.root,bg='#F5E6D3',relief='ridge',bd=4,width=360,height=80)
        self.creation_time = time.time() * 1000

        # 创建姓名标签并放置
        self.name_label = tk.Label(self.text_box,text=self.name,font=('Segoe UI',10,'bold'),bg='#F5E6D3',fg='black')
        self.name_label.place(relx=0,rely=0,anchor='nw')

        # 创建继续提示标签
        self.continue_label = tk.Label(self.text_box,text= '▼',font=('Arial',8,'bold'),fg='#8B4513',bg='#F5E6D3',cursor='hand2')

    def _animation_continue_blink(self):
        '''提示标签闪烁'''
        if not self.is_typing and self.text_box.winfo_viewable():
                self.continue_label.config(fg='#D2691E' if self.continue_label.cget('fg') == '#8B4513' else '#8B4513')
                self.root.after(640, self._animation_continue_blink)
        else:
            self.continue_label.place_forget()

    def display_text_box(self):
        self.text_box.place(relx=0.5,rely=0.9,anchor='s',width=320,height=80)

    def hid_text_box(self):
        self.text_box.place_forget()


    def _on_click(self,event=None):
        if self.text_type == 1:
            current_time = time.time() * 1000

            if current_time - self.creation_time < 60:  # 防止单次点击触发双次操作
                return
            
            if current_time - self.last_click_time >= 120: # 防止误触
                self.last_click_time = current_time
                if self.is_typing:
                    # 立即完成当前文本
                    self.char_index = len(self.current_text)
                    displayed_text = self.current_text
                    self.text_label.config(text=displayed_text)
                    # 取消后续打字动画
                    if self.task_id:
                        self.root.after_cancel(self.task_id)
                        self.task_id = None
                    self.is_typing = False
                    self.continue_label.place(relx=1,rely=1,anchor='se')
                else:
                    # 进入下一文本
                    self._to_next_text()

    def prompt_appears(self,text):
        '''提示文字'''
        self.text_label.config(text=text)
        self.steps = 20
        self.target_rely = 0.05
        self._animation_prompt_drop(0,self.location[1],0)
        

    def _animation_prompt_drop(self,step,start_rely,end_rely):
        '''提示文字下落动画'''
        if step <= self.steps:
            # 线性下落
            progress = step / self.steps
            current_rely = start_rely + (end_rely - start_rely) * progress
                
            self.text_label.place(
                relx=self.location[0], 
                rely=current_rely, 
                anchor=self.location[2]
                )
                
            self.task_id = self.root.after(16, lambda: self._animation_prompt_drop(step+1,start_rely,end_rely))
        else:
            if start_rely < end_rely:
                self._animation_prompt_blink(0)
            else:
                self.text_label.destroy()


    def _animation_prompt_blink(self,step):
        '''提示文字闪烁动画'''
        if step <= 4:
            self.text_label.config(bg=['gold','lightyellow'][step%2])
            self.task_id = self.root.after(240,lambda:self._animation_prompt_blink(step+1))
        else:
            self._animation_prompt_drop(0,0,self.location[1])



'''文本样式与位置(中间浮现、文本框内文字、提示文字)'''
TEXT_STYLE = ( [24,320,'center','white','black'],
               [10,280,'left','#2C1810', '#F5E6D3'],
               [8,0,'left','#8B4513','gold'] )

TEXT_LOCATION = ( [0.5,0.375,'center'],
                  [0.5,0.5,'center'],
                  [0.0,-0.04,'nw'] )