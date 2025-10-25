import random

from constants import GIFT_EFFECTS, DIALOGUES


class Character:
    def __init__(self, name: str, role: str, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self) -> bool:
        print(f"你正在和{self.name}对话...")

        dialogue = random.choice(DIALOGUES[self.name])
        print(f"{self.name}：『{dialogue["text"]}』")

        choice = input(f"1. {dialogue["optionA"]}\n"
                       f"2. {dialogue["optionB"]}\n"
                       f"请选择：")
        if choice == "1":
            print(f"你：『{dialogue["optionA"]}』")
            self.change_affinity(5)
            return True
        else:
            print(f"你：『{dialogue["optionB"]}』")
            self.change_affinity(-5)
            return False

    def give_gift(self, gift: str):
        print(f"你送给 {self.name} 一份 {gift}。")
        gift_effect = GIFT_EFFECTS[gift]
        if self.name in gift_effect:
            self.change_affinity(gift_effect[self.name])
        else:
            self.change_affinity(gift_effect["default"])

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


