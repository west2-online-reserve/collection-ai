from typing import Optional, Dict

from story import DIALOGUES, GIFT_EFFECTS

class Character:
    def __init__(self, name:str, role:str,  affinity: int = 0) ->None :
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogues = DIALOGUES.get(self.name, [])
        if not dialogues:
            self.change_affinity(5)
            return
        idx: int = (self.affinity // 20) % len(dialogues)
        node = dialogues[idx]
        print(f"{self.name}: {node['text']}")
        print(f"A : {node['optionA']}")
        print(f"B : {node['optionB']}")
        choice:str = input("请选择 A/B：").strip().upper()
        if choice == "A":
            self.change_affinity(10)
        elif choice == "B":
            self.change_affinity(5)
        else:
            self.change_affinity(0)


    def give_gift(self, gift: str):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        effect: Optional[Dict] = GIFT_EFFECTS.get(gift)
        if not effect:
            print(f"{self.name}对{gift}表示疑惑")
            self.change_affinity(0)
            return
        value: int = effect.get(self.name, 0)
        if value > 0:
            print(f"{self.name} 对 {gift} 表示 happy")
        elif value < 0:
            print(f"{self.name} 对 {gift} 表示 sad")
        else:
            print(f"{self.name} 对 {gift} 表示 indifferent")
        self.change_affinity(value)


    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化由 {self.affinity - value} -> 当前好感度：{self.affinity}")


    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False
