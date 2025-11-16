from typing import Optional,List,Dict
import random
from constant import DIALOGUES, GIFT_EFFECTS

class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name       # 角色名
        self.role: str = role       # 角色设定
        self.affinity: int = affinity  # 好感度

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        
        dialog: Optional[List[Dict[str, str]]] = DIALOGUES.get(self.name)
        if dialog:
            entry: Dict[str, str] = random.choice(dialog)
            print(self.name + "：" + entry["text"])
            print("A）" + entry["optionA"])
            print("B）" + entry["optionB"])
            choice: str = input("请选择 A 或 B（大小写均可）：").strip().upper()
            
            self.change_affinity(5)  # 基础好感度变化
            if choice == "A":
                self.change_affinity(5) 
            else:
                if random.randint(0, 1) == 0:
                    self.change_affinity(-5)
        else:
            self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        effect: Optional[Dict[str, int]] = GIFT_EFFECTS.get(gift)
        if effect is None:
            print("系统：没有这种礼物，或礼物没有效果。")
            return
        # 优先取角色专属效果，否则取“default”，最后默认0
        value: int = effect.get(self.name, effect.get("default", 0))
        self.change_affinity(value)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False