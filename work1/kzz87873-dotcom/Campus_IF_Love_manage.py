import json


class Character:
    def __init__(self, name: str, role: str, affinity: int=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.dialogues_index = 0

    def talk(self, dialogues: list[dict[str, str]]) -> None:
        if self.dialogues_index >= len(dialogues):
            print(f'{self.name}:无新对话')
            return

        print(f"你正在和{self.name}对话...")
        current_dialogues = dialogues[self.dialogues_index]
        print(f'{self.name}:{current_dialogues["text"]}')
        Choice = input(f'1.{current_dialogues["optionA"]} 2.{current_dialogues["optionB"]}\n 请选择：')
        if Choice == '1':
            self.change_affinity(10)
        if Choice == '2':
            self.change_affinity(5)
        else:
            print('输入无效值，好感度不变')

        self.dialogues_index += 1


    def give_gift(self, gift: str, gift_effects: dict[str, dict[str, int]]) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        particular_effect = gift_effects.get(gift)
        if particular_effect is None:
            print("系统：没有这种礼物，或礼物没有效果。")
            return
        value = gift_effects.get(gift,{}).get(self.name, 0)
        self.change_affinity(value)
        pass

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False

class SaveManager:

     @staticmethod
     def save_game(data: dict, save_path: str = "game_save.json") -> None:
         with open(save_path, 'w', encoding='utf-8') as func:
             json.dump(data, func, ensure_ascii=False, indent=2)
         print("\n存档成功！")

     @staticmethod
     def load_game(save_path: str = "game_save.json") -> dict | None:
         try:
             with open(save_path, 'r', encoding='utf-8') as f:
                 return json.load(f)
         except FileNotFoundError:
             print("\n暂无存档文件！")
             return None