import Text_Framework as tf
import tkinter as tk



class Character:
    '''角色属性与相关操作'''

    text_talk = None

    def __init__(self, name, role, root,affinity=0):
        self.name = name
        self.role = role
        self.attribute = {"affinity":affinity}
        self.root = root


    def talk(self, progress, next_step, variants=0,index=0):
        Character.text_talk = tf.TextFramework(self.root,1,self.name)

        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        Character.text_talk.text_finished = next_step  # 为实例对象设置回调函数（下一步操作）
        
        # 获取对话内容
        current_phase = SCRIPT.get(progress[0])
        current_operation = current_phase[progress[1]]
        dialogues_content = current_operation.get("text")
        if type(dialogues_content[0]) == list:  # 检测是否有差分
            dialogues_content = dialogues_content[variants]

        Character.text_talk.animation_text(dialogues_content,index)
        
    def talk_finished(self):
        self.change_affinity(5)


    def give_gift(self):
        pass

    def give_gift_finished(self, gift):
        text_gift = tf.TextFramework(self.root,2)
        text_gift.prompt_appears(f"🎁 你送给 {self.name} 一份 {gift}")
        value = GIFT_EFFECTS.get(gift)
        self.root.after(4040,lambda:self.change_affinity(value))

        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        

    def change_affinity(self, value):
        self.attribute["affinity"] += value
        text_affinity = tf.TextFramework(self.root,2)
        text_affinity.prompt_appears(f"💖 {self.name} 的好感度 {value:+} >> {self.attribute['affinity']}") 

    def check_ending(self):
        if self.attribute["affinity"] >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False



class Player(Character):
    '''玩家属性'''

    def __init__(self, name, role, root):
        self.name = name
        self.role = role
        self.attribute = {"key":
                          {"rejection 1":[]}
                          }
        self.root = root



class Others:
    '''其他操作'''

    text_framework = None

    @classmethod
    def narration(cls,root, progress, next_step=None,variants=0,index=0):
        # 获取旁白内容
        current_phase = SCRIPT.get(progress[0])
        current_operation = current_phase[progress[1]]
        narration_content = current_operation.get("text")
        if type(narration_content[0]) == list:  # 检测是否有差分
            narration_content = narration_content[variants]
        
        try:
            narration_type = ["on_game_page","on_text_box"].index(current_operation.get("subtype"))  # 尝试获取旁白类型
        except:
            narration_type = 1
        
        cls.text_framework = tf.TextFramework(root, narration_type)  # 创建文本实例   
        if next_step:
            cls.text_framework.text_finished = next_step  # 设置回调

        cls.text_framework.animation_text(narration_content,index)  # 执行动画

    @staticmethod
    def choose(progress, *next_step, variants=0):
        '''处理选择事件'''
        from System_Control import Pages

        # 获取选项内容
        current_phase = SCRIPT.get(progress[0])
        current_operation = current_phase[progress[1]]
        option_content = current_operation.get("text")
        if type(option_content[0]) == list:  # 检测是否有差分
            option_content = option_content[variants]

        def choose_finished(choice):
            if choice == 'A' and next_step[0]:
                next_step[0]()
            if choice == 'B' and next_step[1]:
                next_step[1]()
            if choice == 'C' and next_step[2]:
                next_step[2]()
            if choice == 'D' and next_step[3]:
                next_step[3]()

        Pages.example.display_option(option_content,choose_finished)  # 显示选项，传入回调





DIALOGUES ={
    "学姐": [
        {"text":["这位新生？要不要来试试？"], "optionA": "主动表现兴趣，拿起一只笔作画。", "optionB": "表示抱歉，没兴趣，转身离开。"},
        {"text": ["你是新来的吧？","画画过吗？"], "optionA": "其实我更擅长写代码，但我愿意试试画画。", "optionB": "画画？那种小孩子的东西我可没兴趣。"},
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
    ],
    "你": [
        {"text": ["……………………","感觉不如原神。","……………………"]}
    ]
}



GIFT_EFFECTS = {
    # 通用 / 默认 值
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"default": -10},  # 所有人 -10
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}


'''游戏剧本'''
SCRIPT = {

    "phase 0":  # 开始剧情
            [
            {"type":"narration","subtype":"on_text_box",
            "text":[
                "你是一名刚刚踏入大学校园的新生。",
                "开学典礼上，作为新生代表的你站在千人礼堂中央。",
                "在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。",
                "消息迅速传开，关于“神秘新生代表”的讨论充斥着整个校园。",
                "但你只是微微一笑，深藏功与名——毕竟，真正的实力还在代码里。"
            ]},

            {"type":"narration","subtype":"on_text_box",
            "text":[
                "收起演讲稿，你漫步在校园林荫道。",
            ]},

            {"type":"talk","character":"你",
            "text":[
                "开学典礼总算结束了……",
                "比起这种场合，还是写代码更自在。",
                "不过刚才的发言效果似乎不错？",
                "那么……接下来该去哪里逛一逛呢？",
            ]},

            {"type":"choose",
            "text":[
                "社团活动楼",
                "学校图书馆",
                "校外咖啡店",
            ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        ["你决定去社团活动楼看一看。"],  # to phase 1r1
                        ["你决定去学校图书馆看一看。"],  # to phase 1r2
                        ["你决定去校外咖啡店写代码。"]  # to phase 1r3
                    ]},

            ],


    "phase 1r1":  # 学姐场景 (from phase 0)
            [
            {"type":"narration","subtype":"on_text_box",
            "text":[
                "你正在社团活动楼逛着，正好路过艺术社团活动室。",
                "一位气质清冷的学姐正在里面作画，周围还有许多其他同学正在负责其他的工作。",
                "似乎是发现了你的存在，她也抬起了头。"
            ]},

            {"type":"talk","character":"学姐",
            "text":[
                "同学。",
                "对绘画有兴趣么？要不要来试试。"
            ]},

            {"type":"choose",
            "text":[
                "主动表现兴趣，拿起一只笔作画",  # to connection(1r1) 1, node = 2
                "摇摇头，打算转身离开"
            ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "你转身，打算离开。",
                        "学姐再度开口，周围的同学们都感到有些惊讶，似乎见到了这位清冷的学姐不一样的一面。"
                    ]},

                    {"type":"talk","character":"学姐",
                    "text":[
                        "你真的……",
                        "不愿意尝试一下么？"
                    ]},

                    {"type":"choose",
                    "text":[
                        "拿起一只笔作画",  # to connection(1r1) 1, node = 5
                        "依旧转身离开"
                    ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "在纵目睽睽下，你扬长而去。",
                        "你又随便逛了一会，回到了林荫道。", # to connection(1r1) 2, node = 6
                    ]},

            {"type":"narration","subtype":"on_text_box",  
            "text":[
                "你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。",  # from connection(1r1) 1, node = 7
                "学姐目光一震，眼神变得格外认真。"
            ]}
            ],

    "phase 1r2":  # 小白场景 (from phase 0)
            [
            {"type":"narration","subtype":"on_text_box",
            "text":[
                "你走进学校图书馆，准备找点资料。",
                "在安静的阅览区，一个娇小的身影正趴在桌子上。",
                "面前堆满了编程书籍，她眉头紧锁，似乎遇到了难题。"
            ]},

            {"type":"talk","character":"小白",
            "text":[
                "呜呜……这个递归到底该怎么写啊……",
                "Python 里的 for 循环，我总是写不对……"
            ]},

            {"type":"choose",
            "text":[
                "主动上前帮忙解题",  # to connection(1r2) 1, node = 2
                "假装没看见，继续找书"
            ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "你正准备转身离开。",
                        "小白突然抬头，泪眼汪汪地看着你。"
                    ]},

                    {"type":"talk","character":"小白",
                    "text":[
                        "同学……你看起来很厉害的样子……",
                        "能不能……教教我？"
                    ]},

                    {"type":"choose",
                    "text":[
                        "耐心讲解算法思路",  # to connection(1r2) 1, node = 5
                        "敷衍几句后离开"
                    ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "你简单应付了几句就匆匆离开。",
                        "走出图书馆，你又回到了林荫道。"  # to connection(1r2) 2, node = 6
                    ]},

            {"type":"narration","subtype":"on_text_box",  
            "text":[
                "你坐下来，耐心地为她讲解算法思路。",  # from connection(1r2) 1, node = 7
                "小白的眼睛逐渐亮了起来，露出了崇拜的表情。"
            ]}
            ],

    "phase 1r3":  # 姐姐场景 (from phase 0)
            [
            {"type":"narration","subtype":"on_text_box",
            "text":[
                "你走进校外咖啡店，找了个安静的角落开始敲代码。",
                "一位气质成熟的姐姐坐在不远处。",
                "她似乎对你的代码很感兴趣，缓缓朝你走了过来。"
            ]},

            {"type":"talk","character":"姐姐",
            "text":[
                "你的代码思路很有趣呢。",
                "能给我讲讲你的实现方法吗？"
            ]},

            {"type":"choose",
            "text":[
                "详细解释代码逻辑",  # to connection(1r3) 1, node = 2
                "保持沉默，继续敲代码"
            ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "你头也不抬，继续专注于自己的代码。",
                        "姐姐轻轻叹了口气，但没有离开。"
                    ]},

                    {"type":"talk","character":"姐姐",
                    "text":[
                        "看来你是个很专注的人呢……",
                        "不过，交流思想有时候比独自钻研更有收获哦。"
                    ]},

                    {"type":"choose",
                    "text":[
                        "开始与她讨论技术",  # to connection(1r3) 1, node = 5
                        "收拾东西准备离开"
                    ]},

                    {"type":"narration","subtype":"on_text_box",
                    "text":[
                        "你收拾好电脑，头也不回地离开了咖啡店。",
                        "回到校园林荫道，你继续思考着刚才的代码。"  # to connection(1r3) 2, node = 6
                    ]},

            {"type":"narration","subtype":"on_text_box",  
            "text":[
                "你开始详细解释你的代码架构和算法选择。",  # from connection(1r3) 1, node = 7
                "姐姐认真倾听，不时提出深刻的问题，你们的讨论越来越深入。"
            ]}
            ],

        "phase 0a1":  # 再次选择
                [
                {"type":"narration","subtype":"on_text_box",
                "text":[
                    "你又回到了校园林荫道。"
                ]},

                {"type":"talk","character":"你",
                "text":[
                    ["在社团活动楼转了一圈，没什么特别感兴趣的……",  # from connection(1r1) 2, node = 2
                    "或许我不太适合这种集体活动。",
                    "算了，再去看看别的地方吧！"],
                    ["图书馆里的资料倒是不少。",  # from connection(1r2) 2
                    "就是感觉对我没什么启发……"
                    "看来得去别的地方找找灵感了。"
                    ],
                    ["咖啡店人有点多，找不到安静的位置!",  # from connection(1r3) 2
                    "影响我写代码的效率……"
                    "还是换个地方看看吧。"
                    ]
                ]},

                {"type":"choose",
                "text":[
                    ["学校图书馆", 
                    "校外咖啡店"],
                    ["社团活动楼", 
                    "校外咖啡店"],
                    ["社团活动楼", 
                    "学校图书馆"]
                ]},

                        {"type":"narration","subtype":"on_text_box",
                        "text":[
                            ["你决定再去社团活动楼看一看。"],  # to phase 1r1
                            ["你决定再去学校图书馆看一看。"],  # to phase 1r2
                            ["你决定再去校外咖啡店写代码。"]  # to phase 1r3
                        ]},

                ],

        "phase 0a2":  # 再再次选择
                [
                {"type":"narration","subtype":"on_text_box",
                "text":[
                    "你又又回到了校园林荫道。"
                ]},

                {"type":"talk","character":"你",
                "text":[
                    ["在社团活动楼转了一圈，没什么特别感兴趣的……",  # from connection(1r1) 2, node = 2
                    "或许我不太适合这种集体活动。",
                    "算了，再去看看别的地方吧！"],
                    ["图书馆里的资料倒是不少。",  # from connection(1r2) 2
                    "就是感觉对我没什么启发……"
                    "看来得去别的地方找找灵感了。"
                    ],
                    ["咖啡店人有点多，找不到安静的位置!",  # from connection(1r3) 2
                    "影响我写代码的效率……"
                    "还是换个地方看看吧。"
                    ]
                ]},

                {"type":"choose",
                "text":[
                    ["社团活动楼"],
                    ["学校图书馆"],
                    ["校外咖啡店"]
                ]},

                        {"type":"narration","subtype":"on_text_box",
                        "text":[
                            ["你决定最后去社团活动楼看一看。"],  # to phase 1r1
                            ["你决定最后去学校图书馆看一看。"],  # to phase 1r2
                            ["你决定最后去校外咖啡店写代码。"]  # to phase 1r3
                        ]},

                ],


        "bad_ending 0":  # 全部拒绝
                [
                {"type":"narration","subtype":"on_game_page",
                "text":[
                    "你拒绝了所有相遇的机会。",
                    "大学四年转瞬即逝……",
                    "你独自一人度过了整个大学生涯。",
                    "虽然代码能力很强，但总觉得缺少了什么。",
                    "或许，有些机会一旦错过就不再……"
                ]},

                {"type":"narration","subtype":"on_game_page",
                "text":[
                    "Bad Ending : 孤独的程序员"
                ]},

                {"type":"narration","subtype":"on_text_box",
                "text":[
                    "三个妹子都不要，你也是神人了。👍",
                    "（点击以结束游戏）",
                ]}
                ],
    }