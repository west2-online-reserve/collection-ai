import json
import os
from typing import Dict,Optional
class Savemanager:
    def __init__(self):
        self.save_file="game_save.json"
        self.quick_save_file="quick_save.json"
    
    #保存
    def save_game(self,game_state:Dict,filename:str=None)->bool:
        if filename is None:
            filename=self.save_file
        try:
            with open(filename,'w',encoding="UTF-8") as f:
                json.dump(game_state,f,ensure_ascii=False,indent=2)
            return True 
        except Exception as e:
            print(f"保存失败：{e}")
            return False

    # 加载
    def load_game(self,filename:str=None)->Optional[Dict]:
        if filename is None:
            filename=self.save_file
        try:
            if not os.path.exists(filename):
                print(f"存档文件{filename}不存在")
                return None
            with open(filename,"r",encoding="UTF_8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载失败：{e}")
            return None
                
    def quick_save(self,game_state:Dict)->bool:
        return self.save_game(game_state,self.quick_save_file) 

    def quick_load(self)->Optional[Dict]:
        return self.load_game(self.quick_save_file)

    def auto_save(self,game_state:Dict)->bool:
        return self.save_game(game_state,self.save_file)       