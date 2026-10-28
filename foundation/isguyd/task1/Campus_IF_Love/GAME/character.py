from data import DIALOGUES
from data import GIFT_EFFECTS
class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        dialogues = DIALOGUES.get(self.name)
        for line in dialogues:
            print(line.get("text"))
            print(line.get("optionA"), end="(选项A)    ")
            print(line.get("optionB"), end="(选项B)    ")
            choice = input("请选择选项以推进剧情：")
            if choice == "A":
                self.change_affinity(5)
            elif choice == "B":
                self.change_affinity(-5)
            else:
                print("无效输入")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        effect = GIFT_EFFECTS.get(gift,{})
        value = effect.get(self.name,effect.get("default"))
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        self.change_affinity(value)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False