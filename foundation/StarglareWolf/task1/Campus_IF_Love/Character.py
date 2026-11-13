from Campus_IF_Love import *
from Campus_IF_Love import DIALOGUES

class Character:
    def __init__(self, name: str, role: str, affinity: int =0) -> None:
        self.name = name
        self.role = role
        self.affinity = affinity
        self.talk_order = 0

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        if self.talk_order < len(DIALOGUES[self.name]):
            print(f"{self.name}:『 {DIALOGUES[self.name][self.talk_order]["text"]} 』")
            while True:
                choice = input(
                    f"1. {DIALOGUES[self.name][self.talk_order]["optionA"]}\n2. {DIALOGUES[self.name][self.talk_order]["optionB"]}\n请选择：")
                if choice == "1":
                    self.change_affinity(5)
                    self.talk_order += 1
                    break
                elif choice == "2":
                    self.change_affinity(-3)
                    self.talk_order += 1
                    break
                else:
                    print("没有这个选项!请重新选择")
        else:
            print("与当前角色对话已结束，敬请期待")

    def give_gift(self, gift:str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        self.change_affinity(GIFT_EFFECTS[gift][self.name])
        pass

    def change_affinity(self, value: int ) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False



GIFT_EFFECTS = {
    # 通用 / 默认 值
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"学姐": -10, "小白": -10, "姐姐": -10},  # 所有人 -10
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}