from text import DIALOGUES, GIFT_EFFECTS


class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self)-> None:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        character_text = DIALOGUES.get(self.name)
        if self.affinity <= 0:
            print(character_text[0].get("text"),"\nA",character_text[0].get("optionA"),"B",character_text[0].get("optionB"))
        elif 0 < self.affinity <= 20:
            print(character_text[1].get("text"),"\nA",character_text[1].get("optionA"),"B",character_text[1].get("optionB"))
        elif 20 < self.affinity <= 40:
            print(character_text[2].get("text"),"\nA",character_text[2].get("optionA"),"B",character_text[2].get("optionB"))
        elif 40 < self.affinity <= 70:
            print(character_text[3].get("text"),"\nA",character_text[3].get("optionA"),"B",character_text[3].get("optionB"))
        elif 70 < self.affinity :
            print(character_text[4].get("text"),"\nA",character_text[4].get("optionA"),"B",character_text[4].get("optionB"))
        answer = input("你的选择：")
        if answer == "A":
            self.change_affinity(5)
        else:
            self.change_affinity(0)


    def give_gift(self, gift : str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        effects = GIFT_EFFECTS.get(gift)
        self.change_affinity(effects.get(self.name,-10))
        pass

    def change_affinity(self, value : int)-> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self)-> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False
