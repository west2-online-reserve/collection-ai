from data import DIALOGUES,GIFT_EFFECTS
import sys
import random
from typing import List, Dict, Optional, Union
class Character:
    def __init__(self:str, name:str, role:str, affinity: int =0)->None:
        self.name:str = name
        self.role :str= role
        self.affinity:int = affinity

    def talk(self)-> None:
        print(f"你正在和{self.name}对话...")
        talk1:List[Dict[str, str]]=DIALOGUES.get(self.name)
        talk2: Dict[str, str]=random.choice(talk1)
        print(f"{self.name}:{talk2["text"]}\n")
        print(f"1.{talk2["optionA"]}\n")
        print(f"2.{talk2["optionB"]}\n")
        choice: str =input("请选择：1 or 2")
        if choice == '1':
           self.change_affinity(5)
        elif choice =='2':
            self.change_affinity(-3)
        else:
            print("无效")
            

    def give_gift(self) ->None:
        print('"鲜花" "编程笔记" "奶茶" "奇怪的石头"\n"精致的钢笔" "可爱玩偶" "夜宵外卖"')
        gift: str=input("请选择：")
        GIFT: List[str]=["鲜花","编程笔记","奶茶","奇怪的石头","精致的钢笔","可爱玩偶", "夜宵外卖"]
        if gift in GIFT :                   
         print(f"你送给 {self.name} 一份 {gift}。")
         gift1: Dict[str, int]=GIFT_EFFECTS.get(gift)
         score: int=gift1.get(self.name,gift1.get("default", 0))
         self.change_affinity(score)
        else :
            print("无效")
        
        

    def change_affinity(self, value)-> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self)-> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False