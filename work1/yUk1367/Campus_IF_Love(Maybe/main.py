import tkinter as tk
import System_Control as sc
import os


"""
由于写了一部分别的功能，并且试图写一些奇怪的功能，所以后面函数没写完（被打
重写了部分函数，或许有点倾向task4（吧
"""


if __name__ == "__main__": 
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 修改当前工作目录到脚本所在目录


    game_screen = sc.Pages(tk.Tk())  # 创建游戏窗口
    game_screen.root.mainloop()  # 保持监听