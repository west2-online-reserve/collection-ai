import time

def pause(t:float):
    time.sleep(t)

class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        #根据好感度不同，设置对话内容
        if self.affinity <= 20:
            turn = 0
        elif self.affinity <= 40:
            turn = 1
        elif self.affinity <= 60:
            turn = 2
        elif self.affinity <= 80:
            turn = 3
        else:
            turn = 4

        print(f"你正在和{self.name}对话...")
        pause(0.5)
        print(f"\n{self.role}：{DIALOGUES[self.name][turn]['text']}")
        optionA = DIALOGUES[self.name][turn]['optionA']
        optionB = DIALOGUES[self.name][turn]['optionB']
        pause(0.5)
        print(f"你回答到 1：{optionA}\n          2：{optionB}")
        pause(0.2)
        choice = input("此时你选择：（1 or 2）")
        #对话的好感度增加设置
        pause(0.5)
        if choice == '1':
            print(f"你说：{optionA}")
            self.change_affinity(15)

        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        self.change_affinity(5)

    def give_gift(self, gift):
        pause(0.5)
        print(f"你送给 {self.name} 一份 {gift}。")
        value = GIFT_EFFECTS[gift][self.name]
        pause(0.5)
        if value >= 10:
            print(f"{self.name}激动地收下了{gift}。")
        elif value > 0:
            print(f"{self.name}开心地收下了{gift}。")
        else:
            print(f"{self.name}疑惑地收下了{gift}。")
        self.change_affinity(value)    
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        pass

    def change_affinity(self, value):
        self.affinity += value
        pause(0.5)
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    #查看好感度的函数
    def show(self):
        pause(0.5)
        print(f"\n你问道：{self.name}，我们现在关系是什么样的？")
        pause(0.5)
        print(f"{self.name}：我和你的好感度目前是{self.affinity}哟。")
        pause(2)

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False

DIALOGUES = {
    "学姐": [
        {"text": "你是新来的吧？画画过吗？", "optionA": "其实我更擅长写代码，但我愿意试试画画。", "optionB": "画画？那种小孩子的东西我可没兴趣。"},
        {"text": "这幅画……构图还算不错，你是随便画的吗？", "optionA": "嗯，我是随手画的，没想到能得到你的认可。", "optionB": "不，我可是很认真的！因为你在看嘛。"},
        {"text": "加入社团后，不能三天打鱼两天晒网。", "optionA": "放心吧，我会认真对待。", "optionB": "社团只是玩玩而已吧？不要太严肃了。"},
        {"text": "我平时比较严格，你会不会觉得我很不好相处？", "optionA": "严格才好啊，我喜欢有目标的人。", "optionB": "嗯……确实有点难相处。"},
        {"text": "今天的社团活动结束了，你要不要和我一起收拾？", "optionA": "当然可以，我来帮你。", "optionB": "算了吧，我还有事要做。"}
    ],
    "小白": [
        {"text": "Python 里的 for 循环，我总是写不对……", "optionA": "来，我教你一个小技巧！", "optionB": "这种简单的东西，你都不会？"},
        {"text": "你看我借了好多书，会不会显得很傻？", "optionA": "不会啊，这说明你很爱学习，很可爱。", "optionB": "嗯……的确有点太贪心了。"},
        {"text": "写代码的时候，我总喜欢喝奶茶……你呢？", "optionA": "我也是！来，下次我请你一杯。", "optionB": "我只喝水，健康第一。"},
        {"text": "你会不会觉得我太依赖你了？", "optionA": "依赖我也没关系，我喜欢这种感觉。", "optionB": "嗯……是有点吧。"},
        {"text": "要不要一起留在图书馆自习？我想有人陪。", "optionA": "好啊，我正好也要复习。", "optionB": "算了，我还是回宿舍打游戏吧。"}
    ],
    "姐姐": [
        {"text": "你也喜欢在咖啡店看书吗？", "optionA": "是啊，这里很安静，很适合思考。", "optionB": "其实我只是随便找个地方坐。"},
        {"text": "研究生的生活，其实没有你想象的那么光鲜。", "optionA": "我能理解，压力一定很大吧。", "optionB": "那我还是不要考研了。"},
        {"text": "你觉得，什么样的人最值得依靠？", "optionA": "稳重冷静的人。", "optionB": "长得好看的人。"},
        {"text": "我常常一个人坐到很晚，你不会觉得孤单吗？", "optionA": "有时候孤单是种享受。", "optionB": "一个人太寂寞了，我受不了。"},
        {"text": "如果有一天，我遇到困难，你会帮我吗？", "optionA": "当然会，你不用一个人扛着。", "optionB": "我大概帮不了你吧。"}
    ]
}



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

