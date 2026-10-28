from story import *

class Character:
    def __init__(self, name, role, affinity= 0, talk_turn= 0, give_gift_turn= 0):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.talk_turn = talk_turn  # learn from AI: __init__只执行一次，所以在此记录对话次数
        self.give_gift_turn = give_gift_turn
    def talk(self):
        """
        根据对话次数不同改变对话走向,5次对话后对话库燃尽，只能通过送礼增加好感度。
        """
        if self.talk_turn >= len(DIALOGUES[self.name]):
            print(f'\n对话事件已结束，接下来去试试送礼吧')
            return

        print(f"\n你正在和{self.name}对话...")
        print(f'\n{self.name}：{DIALOGUES[self.name][self.talk_turn]["text"]}')
        print(f'1.{DIALOGUES[self.name][self.talk_turn]["optionA"]}\n2.{DIALOGUES[self.name][self.talk_turn]["optionB"]}')
        choice = input('\n你选择...：')
        if choice == '1':
            print(f'{self.name}感到开心，好感度增加了')
            self.change_affinity(10)
        else:
            print(f'{self.name}：这样啊……')
            print('\n(我是不是说错话了？)')
            self.change_affinity(-5)
        self.talk_turn += 1

    def give_gift(self):
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        if self.give_gift_turn >= 5:
            print('\n买太多礼物了，没钱力\n试试对话吧')
            return

        print(f'\n这些礼物感觉不错:{list(GIFT_EFFECTS.keys())}')
        gift = input(f'\n送什么礼物给{self.name}呢：')
        print(f"你送给 {self.name} 一份 {gift}。")

        value: int = 0
        if gift in GIFT_EFFECTS:
            effects_dict = GIFT_EFFECTS[gift]
            if self.name in effects_dict:
                value = effects_dict[self.name]
            elif 'default' in effects_dict:
                value = effects_dict['default']
                print('\n奇怪的石头是错的……')
            if gift == '夜宵外卖':
                if self.name == '学姐':
                    print('\n学姐似乎不爱吃夜宵')
                elif self.name == '姐姐':
                    print('\n姐姐似乎不爱吃夜宵')
        else:
            print(f'{gift}好像不怎么受欢迎')
        self.change_affinity(value)
        self.give_gift_turn += 1

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 60:
            print(f"\n恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        elif self.talk_turn >= len(DIALOGUES[self.name]) and self.give_gift_turn >= 5:
            print(f'\n大学四年过去了，你和 {self.name} 感情上似乎没什么进展，不过做朋友也挺好的\n你进入了单身线')
            return True
        return False