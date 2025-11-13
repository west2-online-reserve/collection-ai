'''逻辑处理，控制剧情走向'''

from typing import Generator,List,Dict,Tuple
import time

class Manage:
    def __init__(self) -> None:
        '''初始化'''
        self.affinity: int = 0
        self.characters:dict = {"s":"学姐","x":"小白","j":"姐姐"}
        self.name: str | None = None

    def give_gift(self, gift: str) -> None:
        '''计算礼物效果'''
        gift_name = GIFT_LIST[int(gift)]
        value = GIFT_EFFECTS[gift_name][self.name]
        print("\n你成功把礼物连同心意一起传递给她了！！！")
        #time.sleep(1)
        self.change_affinity(value)
        if value < 0: print(f"{self.name}不喜欢这个礼物……")

    def change_affinity(self, value: int) -> None:
        '''更改好感度'''
        self.affinity += value
        print(f"{self.name}的好感度变化{value}\n当前好感度为{self.affinity}")

    def check_affinity(self) -> None:
        '''输出当前好感度'''
        print(f"当前{self.name}对你的好感度为{self.affinity}")

    def choice_warning(self) -> None:
        '''提示玩家输入有误'''
        print("\n没有这个选项,请重新选择")
        if self.name != None: self.change_affinity(-1)

    def check_ending(self) -> bool:
        '''检查是否可以进入结局'''
        if self.affinity >= 100:
            print(f"{self.name}的好感度已经满了，你成功进入结局。")
            return True
        return False
    
    def check_state(self,state: dict, ans: str | None) -> Tuple[str | None, str | None]:
        '''处理状态，发出下一步的信号'''

        #进入下一个节点
        if state["entry"] == None:
            return "next_node",None
        
        #进入隐藏结局
        elif state["entry"] == "stupid":
            return "entry_end","se"
        
        #判断进入True_Ending或Bad_Ending
        elif state["entry"] == "end":
            print("\n你做了一件正确的事！")
            self.change_affinity(10)
            if self.check_ending():
                return "entry_end","te"
            else:
                return "entry_end","be"
        
        #进入选择节点
        elif state["entry"] == "special":
            return "entry_special","000"
        
        #查看好感度
        elif state["entry"] == "affinity":
            self.check_affinity()
            return "entry_special","000"

        #根据玩家选择，进入下一个节点
        elif state["entry"] == "choice":
            if ans in ["1","2","3","4"]:
                return "entry_special",ans.zfill(3)
            else:
                self.choice_warning()
                return "try_again",None
        
        #根据玩家选择的礼物，并再次进入选择节点
        elif state["entry"] == "gift":
            if ans in ["1","2","3","4","5","6","7"]:
                self.give_gift(ans)

                #判断是否进入Good_Ending
                if self.check_ending():
                    return "entry_end","ge"
                
                return "entry_special","000"
            else:
                self.choice_warning()
                return "try_again",None
        
        #进入角色线
        elif state["entry"] in self.characters:
            if ans not in ["a","b"]:
                self.choice_warning()
                return "try_again",None
            elif state["ans"] == ans:
                self.name =self.characters[state["entry"]]
                return "entry_route",state["entry"]
            else:
                return "next_node",None
        
        #根据玩家的选项，更改角色状态
        elif state["entry"] == "question":
            if ans not in ["a","b"]:
                self.choice_warning()
                return "try_again",None
            elif ans == state["ans"]:
                print("\n你做了一件正确的事！")
                self.change_affinity(10)

                #判断是否进入Good_Ending
                if self.check_ending():
                    return "entry_end","ge"
            else:
                print("\n你情商是负数吗？？？")
                self.change_affinity(-5)
            return "entry_special","000"

#礼物相关数据
GIFT_LIST = [None,"鲜花","编程笔记","奶茶","奇怪的石头","精致的钢笔","可爱玩偶","夜宵外卖"]

GIFT_EFFECTS = {
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"学姐": -10,"小白":-10,"姐姐":-10},
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}