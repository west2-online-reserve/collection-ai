import random
from typing import Dict,List,Optional
from data import DIALOGUES,GIFT_EFFECTS


class Character:
    def __init__(self, name:str, role:str, affinity:int=0)->None:
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self)->None:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogue_pool:List[Dict[str,str]]=DIALOGUES.get(self.name,[])
        if dialogue_pool:
            entry:Dict[str,str]=random.choice(dialogue_pool)
            print(self.name+":"+entry["text"])
            print("A)"+entry["optionA"])
            print("B)"+entry["optionB"])
            choice:str=input("请选择A或者B(大小写都可以)：").strip().upper()
            self.change_affinity(5)
            if choice=="A":
                self.change_affinity(5)
            elif choice=="B":
                self.change_affinity(-5)
            else:
                print("无效输入，使用默认回应。")
        else:
            print(f"错误：角色{self.name}没有对话数据！")



    def give_gift(self, gift: str)->None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        effects:Optional[Dict[str,int]]=GIFT_EFFECTS.get(gift)
        if effects is None:
            print(f"{self.name}疑惑地看着这份{gift}，似乎不认识这是什么。")
            print("好感度无变化。")
            return
        value=effects.get(self.name,effects.get("default",0))
        self.change_affinity(value)

        if self.name=="学姐":
            if value>0:
                if value>=15:
                    print("学姐：【这份礼物太用心了，我很喜欢！】")
                elif value>=10:
                    print("学姐：【谢谢，这份礼物很适合我。】")
                else:
                    print("学姐：【谢谢你的礼物。】")
            elif value<0:
                print("学姐：【这份礼物不太合适我......】")
            else:
                print("学姐：【谢谢。】")

        if self.name=="小白":
            if value>0:
                if value>=15:
                    print("小白：【哇！是送给我的吗？！我好喜欢！】")
                elif value>=10:
                    print("小白：【谢谢，这份礼物很可爱。】")
                else:
                    print("小白：【谢谢你的礼物。】")
            elif value<0:
                print("小白：【谢谢你好的好意，不过不太合适，不用啦。】")
            else:
                print("小白：【谢谢。】")

        if self.name=="姐姐":
            if value>0:
                if value>=15:
                    print("姐姐：【这份礼物很有品味，我喜欢。】")
                elif value>=10:
                    print("姐姐：【很不错的礼物，谢谢你的心意。】")
                else:
                    print("姐姐：【谢谢你的礼物。】")
            elif value<0:
                print("姐姐：【抱歉，这个我不太需要。】")
            else:
                print("姐姐：【谢谢。】")


    def change_affinity(self, value: int)->None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self)->bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False