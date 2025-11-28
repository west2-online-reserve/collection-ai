import sys
import random
from Data import DIALOGUES, GIFT_EFFECTS
from typing import List, Dict, Any, Optional, Union #类型标注

class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None: #箭头之后为输出类型
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self) -> None:
        print(f"你正在和{self.name}对话……")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        dialogue_box: List[Dict[str, Any]] = DIALOGUES.get(self.name, [])

        attempt: Dict[str, Any] = random.choice(self.select_dialogues_by_affinity(dialogue_box))
        print(f"{self.name}：{attempt["text"]}")
        print("\nA." + attempt["optionA"])
        print("B." + attempt["optionB"])

        choice: str = input()

        self.process_dialogue_choice(attempt, choice)

    def select_dialogues_by_affinity(self, dialogue_box: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.affinity < 30:
            result: List[Dict[str, Any]] = []
            for dialogue in dialogue_box:
                if not dialogue.get("high_affinity_only", False):
                    result.append(dialogue)
            return result
        elif self.affinity > 70:
            return dialogue_box
        else:
            return dialogue_box
            
    def process_dialogue_choice(self, attempt: Dict[str, Any], choice: str) -> None:
        base_change = 5
        self.change_affinity(base_change)
        if choice == "A":
            self.change_affinity(attempt.get("optionA_effect", 5))
        else:
            self.change_affinity(attempt.get("optionB_effect", -5))

    def give_gift(self, gift: str) -> None:
        print(f"你送给{self.name}一份{gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        effects: Optional[Dict[str, int]] = GIFT_EFFECTS.get(gift)
        value: int = effects.get(self.name, effects.get("default"))
        self.change_affinity(value)
        pass

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name}的对你的好感度变化了{value}。当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和{self.name}的故事进入了结局线！")
            return True
        return False