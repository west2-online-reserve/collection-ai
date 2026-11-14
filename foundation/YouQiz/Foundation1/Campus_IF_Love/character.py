from typing import Dict, List
from story import DIALOGUES, GIFT_EFFECTS

class Character:
    def __init__(self, name: str, role: str, affinity: int=0):
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogs_list: List[Dict[str, str]] = DIALOGUES.get(self.name, [])
        for dialog in dialogs_list:
            print(f"\n{self.name}：「{dialog["text"]}」")
            print(f"选项 A：「{dialog["optionA"]}」")
            print(f"选项 B：「{dialog["optionB"]}」")

            opt = input("请输入选项A/B\n").strip()
            while opt not in ("A", "B"):
                opt = input("输入只能是A/B，请重新输入：").strip()
        self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        gift_data = GIFT_EFFECTS.get(gift, {})
        if "default" in gift_data:
            val = gift_data["default"]
        else:
            val = gift_data.get(self.name)
        self.change_affinity(val)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        if self.affinity < 0:
            self.affinity = 0
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False

    def to_save_dict(self) -> Dict:
        """导出存档字典，用于json序列化"""
        return {
            "name": self.name,
            "role": self.role,
            "affinity": self.affinity
        }

    @staticmethod
    def from_save_dict(data: Dict):
        """从存档字典恢复角色实例"""
        character = Character(name=data["name"], role=data["role"], affinity=data["affinity"])
        return character