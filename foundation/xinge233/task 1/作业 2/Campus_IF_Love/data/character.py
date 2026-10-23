import typing
import random
from .config import *
class Character:
    def __init__(
        self,
        name: typing.Literal["学姐", "小白", "姐姐"],
        role: typing.Literal[
            "社团里的艺术少女", "课堂上的元气同学", "食堂里的温柔姐姐"
        ],
        affinity=0,
    ):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.talk_times = 1
    def talk(self) -> None:
        self.True_answer = "1"
        self.False_answer = "2"
        print(talk_counter.format(self=self))
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        if self.talk_times >= 6 and self.affinity < 90:
            print(random.choice(random_talk[self.name]))
            self.change_affinity(5)
            self.talk_times += 1
        elif self.talk_times >= 6 and self.affinity >= 90:
            print(suki_talk)
            self.change_affinity(5)
            self.talk_times += 1
        else:
            print(DIALOGUES[self.name][self.talk_times - 1]["text"])
            print(give_choice)
            print("1:" + DIALOGUES[self.name][self.talk_times - 1]["optionA"])
            print("2:" + DIALOGUES[self.name][self.talk_times - 1]["optionB"])
            while True:
                self.talk_answer = input(respond_talk)
                if self.talk_answer == self.True_answer:
                    print(happy_talk.format(self=self))
                    self.change_affinity(5)
                    self.talk_times += 1
                    break
                elif self.talk_answer == self.False_answer:
                    print(unhappy_talk.format(self=self))
                    self.change_affinity(-1)
                    self.talk_times += 1
                    break
                elif self.talk_answer == "answer":
                    print(self.True_answer)
                else:
                    print(wrong_output[self.name])

    def give_gift(self) -> None:
        while True:
            print(decide_gift)
            gift_choose=input(choose_by_number)
            
            if gift_choose == "8":
                print("你选择了不送礼物。")
                break
            elif gift_choose in choose_gift:
                
                self.gift_name=choose_gift[gift_choose]
                print(gift_tips.format(present=self.gift_name, name=self.name))
                print(GIFT_EFFECTS[self.gift_name][self.name][1])
                self.change_affinity(GIFT_EFFECTS[self.gift_name][self.name][0])
                break
            else:
                print(wrong_output[self.name])

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(affinity_change.format(self=self, value=value))

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(ending.format(self=self))
            return True
        return False