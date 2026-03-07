import random
from data import DIALOGUES, GIFT_EFFECTS
class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        words = DIALOGUES.get(self.name)
        if words:
            say = random.choice(words)
            print(say["text"])
            print("A:" + say["optionA"])
            print("B:" + say["optionB"])
            choice = input("请选择 A 或 B：")
            if choice == "A":
                self.change_affinity(10)
            elif choice == "B":
                self.change_affinity(-5)
        self.change_affinity(5)

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        effect = GIFT_EFFECTS.get(gift)
        if effect:
            if gift == "奇怪的石头":
                print("不要送这种奇奇怪怪的东西了！",self.name,"感到很疑惑觉得你是个怪人")
                self.change_affinity(-10)
            else:
                value = effect.get(self.name)
                if value:
                    self.change_affinity(value)
                else:
                    self.change_affinity(effect.get("default", 0))

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")