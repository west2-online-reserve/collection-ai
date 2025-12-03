from typing import Dict, List, Optional, Union
from story import DIALOGUES, GIFT_EFFECTS

class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role  # 身份
        self.affinity: int = affinity  # 亲密值

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        index: int = self.affinity // 20
        if index >= 5:
            index = 4
        elif index < 0:
            index = 0

        dialogs: List[Dict[str, str]] = DIALOGUES.get(self.name, [{}])
        print(dialogs[index].get("text", "暂无对话内容"))
        print("A: ", dialogs[index].get("optionA", "暂无对话内容"))
        print("B:", dialogs[index].get("optionB", "暂无对话内容"))
        print("这种情况下，你的选择是：   (请输入A或B)")

        while True:  # 预防错误输入
            ans: str = input().strip().upper()  # 转化大写，消除空白
            if ans in ["A", "B"]:
                break
            print("输入无效，请重新输入A或B：")

        if ans == "A":
            print(f"{self.name}似乎很高兴，亲密度上升了！")
            self.change_affinity(20 + index)
        else:
            print(f"{self.name}似乎不太满意...")
            self.change_affinity(-5 - index)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        gift_dict: Dict[str, int] = GIFT_EFFECTS.get(gift, {})
        value: int = gift_dict.get(self.name, gift_dict.get("default", 0))
        if value <= 0:
            print(f"情况好像不太妙，下次还是不要送{self.name}{gift}了...")
        else:
            print(f"{self.name}很开心，亲密度上升了！")
        self.change_affinity(value)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化了{value}点 ->  当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False