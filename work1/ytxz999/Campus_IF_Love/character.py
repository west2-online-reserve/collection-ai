import sys
import random
import time
from typing import Optional,Dict,List
from story import DIALOGUES,GIFT_EFFECTS

class Character:
    def __init__(self, name: str, role: str, affinity: int=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        #随机触发npc对话
        #get()得到字典中的对话，若找不到返回空列表
        dialogue_pool: List[Dict[str, str]] = DIALOGUES.get(self.name, [])
        if dialogue_pool:
            entry = random.choice(dialogue_pool)
            print(self.name,":", entry["text"])
            print("你想对对方说什么:")
            print("A:",entry["optionA"])
            print("B:",entry["optionB"])
            print("C: 什么都不说，只是看着她")
            Option: str   = input("请输入你的选项").strip().upper()
            if Option == "A":
                self.change_affinity(5)
            elif Option == "B":
                self.change_affinity(-5)
            elif Option == "C":
                print("我的脸上有什么东西吗（一脸疑惑）")
                print("A:", entry["optionA"])
                print("B:", entry["optionB"])
                print("C: 还是什么都不说，只是看着她")
                Option1: str = input("请输入你的选项").strip().upper()
                if Option1 == "A":
                    self.change_affinity(5)
                elif Option1 == "B":
                    self.change_affinity(-5)
                #好感度清空
                elif Option1 == "C":
                    print("我和你说话呢，你尔多龙吗")
                    print("触发隐藏结局----------听不见")
                    self.change_affinity(-100)
        else:
            print("对方：我喜欢你")
            time.sleep(2)
            print("你为啥跟我直接表白啊?!嘎啦game里不是这样!你应该多跟我聊天，然后提升我的好感度。"
                  "偶尔给我送送礼物，然后在那个特殊节日时候跟我有特殊互动。最后在某个我内心神秘事件中.向我表白，我同意跟你在一起，然后我给你看我的特殊CG啊。"
                  "你怎么直接上来跟我表白!?嘎啦game里根本不是这样!我不接受!!")
            self.affinity = 0
            time.sleep(1)
            print("触发隐藏结局-----------旮旯给木大神")
            self.change_affinity(-100)


    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        gift_effect: Dict[str, int] = GIFT_EFFECTS.get(gift, {})  # 先按礼物名取字典
        value: int = gift_effect.get(self.name, gift_effect.get("default", 0))
        if value > 0:
            reactions: List[str] = [
                "哇，谢谢你！我很喜欢这份礼物～",
                "太惊喜了！这正是我想要的！",
                "你真懂我，太开心啦！"
            ]
            print(f"{self.name}: {random.choice(reactions)}")
        elif value < 0:
            reactions: List[str]  = [
                "这礼物...有点特别呢（尴尬笑）",
                "呃...谢谢你的心意，但我不太需要这个。",
                "这是什么呀？我不太喜欢呢..."
            ]
            print(f"{self.name}: {random.choice(reactions)}")
        else:
            reactions: List[str]  = [
                "谢谢你的礼物，我收下啦～",
                "心意收到啦，谢谢！",
                "感谢馈赠～"
            ]
            print(f"{self.name}: {random.choice(reactions)}")
        self.change_affinity(value)  # 调用好感度修改方

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False