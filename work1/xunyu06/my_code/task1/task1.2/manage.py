import random
from story import DIALOGUES,GIFT_EFFECTS
from typing import Dict,List,Any,Optional
class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.dialogue_progress=0

    def talk(self):
        print(f"你正在和{self.name}对话...")
        Total_dialogue = DIALOGUES.get(self.name, [])
        
        if not Total_dialogue:
            print("暂无可用对话")
            return
            
        if self.dialogue_progress >= len(Total_dialogue):
            self.dialogue_progress = 0  

        dialogue = Total_dialogue[self.dialogue_progress]
        dialogue:Dict[str,str]=Total_dialogue[self.dialogue_progress]
        print(f'\n{dialogue['text']}')
        print(f'\nA:{dialogue['optionA']}')
        print(f'\nB:{dialogue['optionB']}')
        self.dialogue_progress+=1

        while True:
            choice = input("请选择你的回答(A/B): ")
            if choice not in ('A','B'):
                print("选择无效，请输入'A'或'B'")
            else:
                if choice == 'A':
                    print(f"{self.name}对你的回答非常满意，对你的好感度增加了！")
                    self.change_affinity(10)
                elif choice =='B':
                    print(f"{self.name}似乎不太高兴...不想要再搭理你了")
                    self.change_affinity(0)
            
                print(f'\n与{self.name}的对话结束')
                break



        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）

    def give_gift(self) -> None:
        gifts = list(GIFT_EFFECTS.keys())
        for i, gift in enumerate(gifts, 1):
            print(f"{i}. {gift}")
        while True:
            try:
                choice=input("请输入礼物编号: ")
                choice = int(choice)
                if choice>(len(gifts)) and choice<1:
                    print("取消了礼物赠送。")
                    return
                if 1<=choice<=len(gifts):
                    selected_gift=gifts[choice-1]
                    break
                else:
                    print("无效的选择，请重新输入。")
            except ValueError:
                print("请输入有效的数字。")
        gift_effects=GIFT_EFFECTS.get(selected_gift, {})
        effect=gift_effects.get(self.name, gift_effects.get("default", 0))
        self.change_affinity(effect)
        if effect>0:
            print(f"{self.name}很喜欢你的礼物！好感度+{effect}")
        elif effect<0:
            print(f"{self.name}不太喜欢这个礼物...好感度{effect}")
        else:
            print(f"{self.name}对这个礼物没什么感觉。")
                            



        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        pass

    def change_affinity(self, value:int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False
   