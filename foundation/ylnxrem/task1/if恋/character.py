from data import DIALOGUES, GIFT_EFFECTS


class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity
        self.count: int = 0

    def talk(self) -> bool:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        self.change_affinity(5)
        dia: list[dict[str, str | list[str]]] = DIALOGUES[self.name]
        if self.count >= len(dia):
            return False
        else:
            print(dia[self.count]["text"])
            print(dia[self.count]["optionA"][0])
            print(dia[self.count]["optionB"][0])
            self.count += 1
        return True

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        try:
            self.change_affinity(GIFT_EFFECTS[gift][self.name])
        except KeyError:
            self.change_affinity(-10)

    def change_affinity(self, value: int | str) -> None:
        self.affinity += int(value)
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False
