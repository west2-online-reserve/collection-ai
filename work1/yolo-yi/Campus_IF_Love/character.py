import random
import dialogue
import gift_effects

class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialog=dialogue.DIALOGUES.get(self.name)
        d=random.choice(dialog)
        print(self.name+' : '+d.get("text"))
        print("A"+' : '+d.get("optionA"))
        print("B"+' : '+d.get("optionB"))
        choice=input("请从 A / B 中选择：")
        if choice=="A":
            self.change_affinity(5)
        else:
            self.change_affinity(-3)
        self.change_affinity(5)

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        gift_exist=gift_effects.GIFT_EFFECTS.get(gift)
        if gift_exist:
            value=gift_exist.get(self.name)
            self.change_affinity(value)
        else:
            print("但是该礼物并没有取得芳心")
            return

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False

