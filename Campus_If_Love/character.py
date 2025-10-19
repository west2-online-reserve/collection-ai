from typing import Dict,List,Optional
class Character:
    def __init__(self,name:str,affinity:int=0):
        self.name:str=name
        self.affinity:int=affinity
        self.dialogue_index:int=0
    
    #对话
    def talk(self,dialogues:List[Dict])->int:
        if self.dialogue_index>=len(dialogues):
            self.dialogue_index=0
        dialogue=dialogues[self.dialogue_index]
        print(f"{self.name}:{dialogue['text']}")
        print(f"A.{dialogue['optionA']}")
        print(f"B.{dialogue['optionB']}")
        choice=input("请选择(A/B):").upper()
        if choice=="A":
            change=10
        elif choice=="B":
            change=-10
        else:
            change=0
            print("无效选择")
        current_affinity=self.affinity
        if isinstance(current_affinity,str):
            try:
                current_affinity=int(current_affinity)
            except ValueError:
                current_affinity=0    
        self.affinity+=change
        self.dialogue_index+=1
        if self.dialogue_index>=len(dialogue):
            self.dialogue_index=0
        return change 

    #送礼物
    def give_gift(self,gift:str)->int:
        gift_effects={
            "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
            "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
            "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
            "奇怪的石头": {"default": -10},  # 所有人 -10
            "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
            "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
            "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
        }
        if gift in gift_effects:
            if self.name in gift_effects[gift]:
                effect=gift_effects[gift][self.name]
            elif "default" in gift_effects[gift]:
                effect=gift_effects[gift]["default"]
            else:
                effect=0
        else:
            effect=0
        self.affinity+=effect

        if effect>0:
            print(f"{self.name}喜欢礼物")
        elif effect<0:
            print(f"{self.name}不喜欢礼物")
        else:
            print(f"礼物对{self.name}无效")
        return effect
    #结局
    def check_ending(self)->bool:
        return self.affinity>=100
    #角色
    def get_status(self)->Dict:
        return{"name":self.name,"affinity":self.affinity,"dialogue_index":self.dialogue_index}
    #加载
    def load_status(self,status:Dict)->None:
        self.affinity=int(status.get("affinity",0))
        self.dialogue_index=int(status.get("dialogue_index",0))                                     