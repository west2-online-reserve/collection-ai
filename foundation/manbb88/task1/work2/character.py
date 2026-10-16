from data import DIALOGUES, GIFT_EFFECTS

class Character:
    def __init__(self, name: str, role: str, affinity: int = 0, cnt: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity
        self.cnt: int = cnt

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        self.cnt += 1
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        idx: int = 0
        if self.cnt == 1:
            idx = 0
        elif self.cnt == 2:
            idx = 1
        elif self.cnt == 3:
            idx = 2
        elif self.cnt == 4:
            idx = 3
        elif self.cnt == 5:
            idx = 4
        else:
            print(f"{self.name}：『别聊了送我点礼物吧。』")
            return

        dialog: dict = DIALOGUES.get(self.name)[idx]
        print(f"{self.name}:{dialog.get('text')}")
        choice: str = input(
            f"请选择回应：\n"
            f"A. {dialog.get('optionA')}\n"
            f"B. {dialog.get('optionB')}\n"
            f"你选择："
        )
        self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        self.change_affinity(GIFT_EFFECTS.get(gift).get(self.name, -10))

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False