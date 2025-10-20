from Campus_IF_Love_Story import DIALOGUES, GIFT_EFFECTS
import random
class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        dialogue_content = DIALOGUES.get(self.name,[])
        if len(dialogue_content) > 0:
            dialogue = random.choice(dialogue_content) #从对话列表里抽一个字典
            print(self.name,":",dialogue["text"])
            print("A)",dialogue["optionA"])
            print("B)",dialogue["optionB"])
            reply = input("请在'A'与'B'中选择一个作为你的回答吧~")
            if reply == "A":
                self.change_affinity(5)
            elif reply == "B":
                self.change_affinity(-5)
            else:
                print("输入无效哦，请输入A或B呢！")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        else:
            self.change_affinity(5)

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        value_list = GIFT_EFFECTS.get(gift)

        if value_list is None:
            print("请认真选择哦，您手上没有这个礼物呢~")
            return
        value = value_list.get(self.name , value_list.get("default", 0)) #默认值为0
        self.change_affinity(value)
        if value > 0:
            print(f"{self.name}惊讶并激动地地收下了礼物：'谢谢你！我很喜欢！'")
        elif value < 0:
            print(f"{self.name}接过了礼物，但看起来不太开心，只是说：'呃呃，谢..谢...?'")
        else:
            print(f"{self.name}平静地收下了礼物：'谢谢。'")

        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        pass

    def change_affinity(self, value):
        self.affinity += value
        if self.affinity <= 0:
            self.affinity = 0
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        if self.affinity == 0:
            print(f"你以高超的低情商话术获得了{self.name}的厌恶，你们的关系走向低谷甚至破裂")
        return False

    