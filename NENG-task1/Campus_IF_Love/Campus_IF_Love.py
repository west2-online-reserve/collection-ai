import sys


class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        dialogues = DIALOGUES[self.name]
        for i in range(len(dialogues)):
            print(f"\n{dialogues[i]['text']}")

            choice1 = input(f"\nA: {dialogues[i]['optionA']}\nB: {dialogues[i]['optionB']}\n请选择：")
            if choice1 == "A":
                print(f"\n{dialogues[i]['replyA']}")
                self.change_affinity(15)
            elif choice1 == "B":
                print(f"\n{dialogues[i]['replyB']}")
                self.change_affinity(-10)
            else:
                print("无效输入，跳过本轮。")
        self.change_affinity(5)

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        effects = GIFT_EFFECTS.get(gift, {})
        value = effects.get(self.name, effects.get("default", 0))
        self.change_affinity(value)

        reply = GIFT_REPLIES.get(gift, {}).get(self.name)
        if reply:
            print(f"\n{reply['text']}")
            print(f"{reply['reply']}")

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target = self.characters["None"]

    def start(self):
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        if not self.scene_senpai():
            if not self.scene_xiaobai():
                if not self.scene_jiejie():
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self):
        print("\n【场景一：社团学姐】")
        print("你路过社团活动室，学姐正拿着画板注意到你。")
        print("学姐：『这位新生？要不要来试试？』")

        choice = input("1. 主动表现兴趣，拿起一只笔作画\n2. 表示抱歉，没兴趣，转身离开\n请选择：")
        if choice == "1":
            print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
            print("学姐目光一震，眼神变得格外认真。你进入【学姐线】！")
            self.current_target = self.characters["学姐"]
            self.story_loop()
            return True
        else:
            print("在众目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            self.current_target = self.characters["小白"]
            print("\n你拉过椅子坐下，三笔两笔把算法拆成图解，连旁边自习的人都探头来看。")
            print("小白盯着草稿纸看了好几秒，小声说：『你讲得比老师还清楚……』")
            print("她默默把你的微信备注改成了『图书馆大神』。你进入【小白线】！")
            self.story_loop()
            return True
        else:
            print("\n你摆摆手：『这种题网上搜一下就有。』说完便走了。")
            print("小白没再叫你。等你下次路过那张桌子时，她已经换了座位。")
            print("你错过【小白线】。")
            return False

    def scene_jiejie(self):
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码，一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来……")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            self.current_target = self.characters["姐姐"]
            print("\n你缓缓低眉，手指还在键盘上没停，语气平淡地开始拆解思路。")
            print("说到关键处你顺手把屏幕转过去，她微微俯身，认真看了几秒。")
            print("姐姐：『……你讲得比我想象中深。』她眼神亮了一下，递来一张名片。")
            print("你进入【姐姐线】！")
            self.story_loop()
            return True
        else:
            print("\n你头也不抬，继续敲代码，仿佛没听见。")
            print("她站了几秒，轻轻说了句『打扰了』，转身离开。")
            print("等你终于抬头时，对面只剩一杯没动过的咖啡。你错过【姐姐线】。")
            return False

    def story_loop(self):
        """角色线主循环"""
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = input("请输入选项：")
            if choice == "1":
                self.current_target.talk()
            elif choice == "2":
                print(f"你决定给{self.current_target.name}送礼物，你决定挑选：")
                gift = input("可选：\"鲜花\",\"编程笔记\",\"奶茶\",\"奇怪的石头\",\"精致的钢笔\",\"可爱玩偶\",\"夜宵外卖\"\n请输入礼物名：")
                if gift in GIFT_EFFECTS:
                    self.current_target.give_gift(gift)
                else:
                    print("没有这个礼物，请重新选择。")
            elif choice == "3":
                print(f"{self.current_target.name}的好感度为：{self.current_target.affinity}")
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            else:
                print("无效输入，请重新选择。")

            if self.current_target.check_ending():
                print("游戏结束。")
                break

DIALOGUES = {
    "学姐": [
        {"text": "你是新来的吧？画画过吗？", "optionA": "其实我更擅长写代码，但我愿意试试画画。", "optionB": "画画？那种小孩子的东西我可没兴趣。",
         "replyA": "学姐挑了下眉：『写代码的？有意思。』她把画笔递过来，『那就用写代码的脑子，试试画画的活。』",
         "replyB": "学姐收回画笔，笑意淡了半分：『哦？那你是来干嘛的。』她没再说话，只是抱着手臂看你。"},
        {"text": "这幅画……构图还算不错，你是随便画的吗？", "optionA": "嗯，我是随手画的，没想到能得到你的认可。", "optionB": "不，我可是很认真的！因为你在看嘛。",
         "replyA": "学姐轻轻哼了一声：『随手画成这样？』她把画纸转了个角度，又看了两眼，『行吧，算你有点意思。』",
         "replyB": "学姐愣了一下，随即别开视线：『……你这人说话倒是直接。』她嘴角没压住，把画纸往你面前推了推，『那认真画的话，能画成什么样？』"},
        {"text": "加入社团后，不能三天打鱼两天晒网。", "optionA": "放心吧，我会认真对待。", "optionB": "社团只是玩玩而已吧？不要太严肃了。",
         "replyA": "学姐点点头：『记住你自己说的。』",
         "replyB": "学姐皱了皱眉：『如果你是这种态度，那还是别加入了。』"},
        {"text": "我平时比较严格，你会不会觉得我很不好相处？", "optionA": "严格才好啊，我喜欢有目标的人。", "optionB": "嗯……确实有点难相处。",
         "replyA": "学姐怔了一下，随即笑了：『你倒是会说话。』",
         "replyB": "学姐沉默了一瞬：『……是吗。』她低下头，继续收拾画具。"},
        {"text": "今天的社团活动结束了，你要不要和我一起收拾？", "optionA": "当然可以，我来帮你。", "optionB": "算了吧，我还有事要做。",
         "replyA": "学姐把一叠画纸递给你：『那就麻烦你了。』",
         "replyB": "学姐摆摆手：『行，那你先走吧。』"}
    ],
    "小白": [
        {"text": "Python 里的 for 循环，我总是写不对……", "optionA": "来，我教你一个小技巧！", "optionB": "这种简单的东西，你都不会？",
         "replyA": "小白眼睛一亮：『真的吗？快教教我！』",
         "replyB": "小白低下头，声音小了下去：『……对不起。』"},
        {"text": "你看我借了好多书，会不会显得很傻？", "optionA": "不会啊，这说明你很爱学习，很可爱。", "optionB": "嗯……的确有点太贪心了。",
         "replyA": "小白脸一红：『可、可爱什么的……』",
         "replyB": "小白把书往怀里收了收：『……是吗。』"},
        {"text": "写代码的时候，我总喜欢喝奶茶……你呢？", "optionA": "我也是！来，下次我请你一杯。", "optionB": "我只喝水，健康第一。",
         "replyA": "小白笑得很开心：『说好了哦！』",
         "replyB": "小白撇撇嘴：『你好无趣哦。』"},
        {"text": "你会不会觉得我太依赖你了？", "optionA": "依赖我也没关系，我喜欢这种感觉。", "optionB": "嗯……是有点吧。",
         "replyA": "小白小声说：『那……我就继续依赖你啦。』",
         "replyB": "小白愣了一下：『……我以后会注意的。』"},
        {"text": "要不要一起留在图书馆自习？我想有人陪。", "optionA": "好啊，我正好也要复习。", "optionB": "算了，我还是回宿舍打游戏吧。",
         "replyA": "小白开心地拉开旁边的椅子：『那你坐这里！』",
         "replyB": "小白有点失望：『哦……那下次吧。』"}
    ],
    "姐姐": [
        {"text": "你也喜欢在咖啡店看书吗？", "optionA": "是啊，这里很安静，很适合思考。", "optionB": "其实我只是随便找个地方坐。",
         "replyA": "姐姐笑了笑：『那我们算是同类。』",
         "replyB": "姐姐点点头，没再多说什么。"},
        {"text": "研究生的生活，其实没有你想象的那么光鲜。", "optionA": "我能理解，压力一定很大吧。", "optionB": "那我还是不要考研了。",
         "replyA": "姐姐看了你一眼：『你倒是挺会体谅人。』",
         "replyB": "姐姐笑了一下：『也没必要因为我一句话就放弃。』"},
        {"text": "你觉得，什么样的人最值得依靠？", "optionA": "稳重冷静的人。", "optionB": "长得好看的人。",
         "replyA": "姐姐若有所思：『……说得不错。』",
         "replyB": "姐姐被逗笑了：『你还真是直白。』"},
        {"text": "我常常一个人坐到很晚，你不会觉得孤单吗？", "optionA": "有时候孤单是种享受。", "optionB": "一个人太寂寞了，我受不了。",
         "replyA": "姐姐轻轻点头：『你比同龄人通透。』",
         "replyB": "姐姐摇摇头：『那你可能还不太习惯独处。』"},
        {"text": "如果有一天，我遇到困难，你会帮我吗？", "optionA": "当然会，你不用一个人扛着。", "optionB": "我大概帮不了你吧。",
         "replyA": "姐姐看着你，眼神柔和了些：『记住你这句话。』",
         "replyB": "姐姐笑了笑：『诚实也是种优点。』"}
    ]
}


GIFT_EFFECTS = {
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"default": -10},
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}

GIFT_REPLIES = {
    "鲜花": {
        "学姐": {"text": "学姐接过花，低头看了看：『……花？你倒是会挑。』",
                 "reply": "『不过下次别破费了。』她嘴上这么说，却把花插进了画筒旁边的瓶子里。"},
        "小白": {"text": "小白眼睛一亮：『哇，好漂亮！是给我的吗？』",
                 "reply": "她抱着花笑得有点不好意思：『我……我都没收过几次花呢。谢谢你！』"},
        "姐姐": {"text": "姐姐接过花，指尖在花瓣上轻轻碰了一下：『还挺会选。』",
                 "reply": "她抬眼笑了笑：『谢谢，我很久没收到花了。』语气很淡，但看得出心情不错。"}
    },
    "编程笔记": {
        "学姐": {"text": "学姐翻了翻：『……整理得还挺细。』",
                 "reply": "『行，这份我收下了。』她把笔记夹进自己的画册里，『以后有问题可以来找我。』"},
        "小白": {"text": "小白捧着笔记，眼睛都亮了：『这、这是你整理的？好全啊！』",
                 "reply": "她小声嘀咕：『比我抄的好多了……』然后认真抬头：『我一定好好看！』"},
        "姐姐": {"text": "姐姐随意翻了两页，动作忽然慢下来：『这思路……是你自己写的？』",
                 "reply": "她合上笔记，认真看了你一眼：『有点东西。这份我留着慢慢看。』"}
    },
    "奶茶": {
        "学姐": {"text": "学姐接过奶茶，吸了一口：『……甜度刚好。』",
                 "reply": "她嘴角微微扬了一下：『你怎么知道我不喜欢太甜。算了，这次算你有心。』"},
        "小白": {"text": "小白捧着奶茶笑得眼睛弯弯：『是我最喜欢的那家！你怎么知道的？』",
                 "reply": "她小口吸了一下，满足地眯起眼：『嘿嘿，今天心情超级好。』"},
        "姐姐": {"text": "姐姐看了看杯子上贴的标签，笑了一声：『连我常点的口味都记住了？』",
                 "reply": "她抿了一口：『不错。看来你比我想的细心。』"}
    },
    "奇怪的石头": {
        "学姐": {"text": "学姐盯着那块石头看了两秒：『……你认真的？』",
                 "reply": "她把石头推回来，语气淡淡：『拿走。我对石头没兴趣。』"},
        "小白": {"text": "小白捧着石头，表情有点茫然：『呃……这个……是礼物吗？』",
                 "reply": "她干笑了一下：『谢、谢谢？我会……好好收着的……』说完悄悄把石头放到了桌角。"},
        "姐姐": {"text": "姐姐捏起石头看了看，似笑非笑：『你这是……在考验我的品味？』",
                 "reply": "她把石头轻轻放回你面前：『心意我领了，石头你留着吧。』"}
    },
    "精致的钢笔": {
        "学姐": {"text": "学姐接过钢笔，翻看了一下笔身：『……这笔不便宜吧。』",
                 "reply": "她握在手里试了试重量，语气认真了半分：『谢谢。我会用的。』"},
        "小白": {"text": "小白小心翼翼地捧着：『好漂亮……这个很贵吧？』",
                 "reply": "她有点不好意思地笑了：『我都不太敢用了……谢谢你，我会珍惜的！』"},
        "姐姐": {"text": "姐姐把钢笔在指间转了一圈：『眼光不错。』",
                 "reply": "她收进包里，抬头笑了笑：『这支笔，我收得心安理得。』"}
    },
    "可爱玩偶": {
        "学姐": {"text": "学姐拎起玩偶看了看：『……你觉得我会喜欢这个？』",
                 "reply": "她嘴上嫌弃，却把玩偶放到了画架边上：『算了，放着也不难看。』"},
        "小白": {"text": "小白一把抱住：『好可爱！！』",
                 "reply": "她抱着玩偶晃了晃，笑得特别开心：『我要把它放床头！谢谢你！』"},
        "姐姐": {"text": "姐姐看着玩偶，忍不住笑出声：『你还挺有意思。』",
                 "reply": "她捏了捏玩偶的耳朵：『行吧，这个我收下了，挺可爱的。』"}
    },
    "夜宵外卖": {
        "学姐": {"text": "学姐看了眼外卖：『……你倒是会挑时间。』",
                 "reply": "她没接，只淡淡说：『我不吃夜宵。你自己吃吧。』"},
        "小白": {"text": "小白闻到香味，眼睛一亮：『是夜宵！你怎么知道我饿了！』",
                 "reply": "她一边拆盒子一边笑：『嘿嘿，那我就不客气啦！』"},
        "姐姐": {"text": "姐姐看了眼外卖，笑着摇头：『这个点送夜宵，你是想让我长胖？』",
                 "reply": "她把外卖推回来：『心意领了，不过这个点我不吃东西。』"}
    }
}


if __name__ == "__main__":
    game = Game()
    game.start()