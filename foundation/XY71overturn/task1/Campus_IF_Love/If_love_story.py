from If_love_manage import DIALOGUES, GIFT_EFFECTS



class Character:
    def __init__(self, name, role, affinity=0) -> None :
        self.name = name
        self.role = role
        self.affinity = affinity
        self.dialogue_times = {}
    def talk(self) -> None :
        print(f"你正在和{self.name}对话...")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        if self.affinity < 20:
            band = "0-19"
        elif self.affinity < 40:
            band = "20-39"
        elif self.affinity < 60:
            band = "40-59"
        elif self.affinity < 80:
            band = "60-79"
        elif self.affinity < 100:
            band = "80-99"
        scenes = DIALOGUES[self.name][band]
        count = self.dialogue_times.get(band, 0)
        scene = min(count, len(scenes) - 1)

        if count == 0:
            print(scenes[scene]["text"])
            print("A.", scenes[scene]["optionA"])
            print("B.", scenes[scene]["optionB"])
            print("C.", scenes[scene]["optionC"])
            option = input("请选择：").upper()
            if option not in ("A","B","C"):
                print("选择无效")
                return
            else:
                print(scenes[scene][f"reply{option}"])
                self.change_affinity(scenes[scene][f"SCORE_{option}"])
                self.dialogue_times[band] = count + 1
        

        if count == 1:
            print(scenes[scene]["text"])
            print("A.", scenes[scene]["optionA"])
            print("B.", scenes[scene]["optionB"])
            print("C.", scenes[scene]["optionC"])
            option = input("请选择：").upper()
            if option not in ("A","B","C"):
                print("选择无效")
                return
            else:
              print(scenes[scene][f"reply{option}"])
              self.change_affinity(scenes[scene][f"SCORE_{option}"])
              self.dialogue_times[band] = count + 1

        if count == 2:
            print(scenes[scene]["text"])
            print("A.", scenes[scene]["optionA"])
            print("B.", scenes[scene]["optionB"])
            option = input("请选择：").upper()
            if option not in ("A","B") :
                print("选择无效")
                return
            else:
                print(scenes[scene][f"reply{option}"])
                self.change_affinity(scenes[scene][f"SCORE_{option}"])
                self.dialogue_times[band] = count + 1  


# 对话简化部分多亏了GPT，它告诉我不必要一个个穷举，而是创建字典引用选项与对话，还可以利用字符串索引直接定位选项。至于文本我也是用GPT生成的
    def give_gift(self, gift) -> None :
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度）并调用 change_affinity（）函数 传入此次好感度变化的数值 value
        if gift in GIFT_EFFECTS:
            print(GIFT_EFFECTS[gift][self.name]["reply"])
            self.change_affinity(GIFT_EFFECTS[gift][self.name]["score"])

        

    def change_affinity(self, value) -> None :
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool :
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            if self.name == "Azazel":
                print("你和 Azazel 都收到了心仪方向的录取通知。她约你回社团活动室，拿出一张画。画上是你初遇时，站在窗边提笔的样子。\nAzazel：『画了很久，一直没拿给你看。』")
                print("你：画得很好。我也想把以后画的东西，第一个给你看。Azazel，我喜欢你。』\nAzazel：『嗯，我知道。』\n她收起画，认真看着你：『我也喜欢你。以后有空，一起画吧。』")
                print("升学后，你们找到一间有宽大窗户的画室。Azazel 摆好画具，把一张纸推到你面前。\nAzazel：『今天画什么？』")
                print("你：『还没想好。』\nAzazel：『那先坐着。灵感来了再画。』\n你打开电脑：『我倒是有个程序的想法——』")
                print("Azazel 把画笔放进你手里。\nAzazel：『先陪我画十分钟。十分钟后，我听你讲。』\n你：『说定了？』\nAzazel：『说定了。』")
            elif self.name == "Apollyon":
                print("你走进图书馆，发现常坐的位置上有两张纸。一张是 Apollyon 刚入学时写满修改痕迹的循环练习，另一张是她独立完成的小项目截图。\nApollyon：『我整理资料时翻到了第一张。那时候我连 for 循环都写不明白。』")
                print("你：『现在你已经会给自己的程序写测试了。』\nApollyon：『因为后来有人肯一遍遍听我问。』\n她把旁边的椅子拉开：『以后我还想和你一起学。不过我也想帮你，不只是等你教我。』")
                print("你：『好。以后有不会的，我们一起研究。我也喜欢和你待在一起。』\nApollyon：『我也是。』\n她戴上耳机，又摘下一边：『那今天先从哪道题开始？』")
                print("图书馆里，你和 Apollyon 各自看着电脑。她忽然把屏幕转向你。\nApollyon：『你看，我找到你刚才那个问题的原因了。』\n你看了一眼：『还真是。你怎么发现的？』")
                print("Apollyon：『先看报错，再一点点试。某人最早教我的。』\n你：『那位老师教得不错。』\nApollyon：『嗯。就是讲起 Python 会忘记控制音量。』")
                print("你们同时看向墙上的“保持安静”，忍着笑把声音压低。两台电脑并排亮着，桌上还放着没看完的书。")
            elif self.name == "???":
                print("店里只剩最后几桌客人。学姐合上书，却没有立刻起身。你们从各自最近的难题，聊到想去的地方，又聊回第一次在这里见面。\n???：『那天我只是想问问你的代码，没想到后来会一直等着和你聊天。』")
                print("你：『我也是。有些事和你说完，好像就没那么乱了。』\n???：『我不需要你每次都替我解决问题。有你愿意听，我就很高兴。』")
                print("你：『我喜欢你，以后遇到问题，我们一起想办法，好吗？』\n???：『嗯，我也喜欢你。』\n她伸出手，等你握住。『不过今晚先想办法赶上末班车。』")
                print("你们与店员道别，走进夜色。路口的风有些凉，她将风衣领口拢好，另一只手仍握着你的手。\n???：『下次来，还是坐靠窗那桌？』")
                print("你：『好。不过你得让我先点完咖啡，再问代码。』\n???：『那可不一定。』\n你们笑着走过路口，开始商量下一次见面的时间。")
            return True
        return False