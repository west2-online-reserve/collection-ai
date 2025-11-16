from typing import Dict, List, Optional

class Character:
    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity
        self.dialogue_count: int = 0
        self.dialogue_index: int = 0

    def talk(self, dialogues: List[Dict[str, str]]) -> None:
        print(f"\n你正在和{self.name}对话...")
        
        if self.dialogue_index >= len(dialogues):
            self.dialogue_index = 0  
            
        dialogue = dialogues[self.dialogue_index]
        print(f"{self.name}：『{dialogue['text']}』")
        print(f"A. {dialogue['optionA']}")
        print(f"B. {dialogue['optionB']}")
        
        choice = input("请选择(A/B): ").upper()
        
        # 根据选择增加不同好感度
        if choice == "A":
            affinity_change = 10
            print("你的回答让对方很开心！")
        elif choice == "B":
            affinity_change = 3
            print("你的回答比较普通...")
        else:
            affinity_change = 0
            print("无效选择，对话草草结束。")
        
        self.change_affinity(affinity_change)
        self.dialogue_count += 1
        self.dialogue_index += 1

    def give_gift(self, gift: str, gift_effects: Dict[str, Dict[str, int]]) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        
        # 获取礼物效果
        effect = 0
        if gift in gift_effects:
            if self.name in gift_effects[gift]:
                effect = gift_effects[gift][self.name]
            elif "default" in gift_effects[gift]:
                effect = gift_effects[gift]["default"]
        
        if effect > 0:
            print(f"{self.name}很喜欢这个礼物！")
        elif effect < 0:
            print(f"{self.name}似乎不太喜欢这个礼物...")
        else:
            print(f"{self.name}对这个礼物没什么特别反应。")
        
        self.change_affinity(effect)

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"\n🎉 恭喜！你和 {self.name} 的故事进入了结局线！")
            print(f"你们的关系达到了新的高度，美好的未来在等待着你们...")
            return True
        return False