import sys
from idlelib.configdialog import changes
from time import sleep


class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.known = False

    def talk(self):
        if not self.known:
            print(f"你正在和{self.name}对话...")
            dialogues = DIALOGUES.get(self.name, [])

            if not dialogues:
                print("暂无对话内容")
                return

            # 遍历所有对话
            for dialogue in dialogues:
                print(f"\n{self.name}：{dialogue['text']}")
                print(f"A. {dialogue['optionA']['text']}")
                print(f"B. {dialogue['optionB']['text']}")

                choice = input("请选择(A/B)：").upper()

                if choice == "A":
                    print(f"你：{dialogue['optionA']['text']}")
                    print(f"{dialogue['optionA']['reaction']}")
                    self.change_affinity(5)  # 选项A增加好感度
                elif choice == "B":
                    print(f"你：{dialogue['optionB']['text']}")
                    print(f"{dialogue['optionB']['reaction']}")
                    self.change_affinity(2)  # 选项B增加较少好感度
                else:
                    print("无效选择，跳过此对话")
                continue
        else:
            print("神金，又聊一次，快去送个礼物争取拿下！")
        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）
        self.known = True

    def give_gift(self, gift):
        print(f"你送给 【{self.name}】一份 {gift}。")
        print(f"【{self.name}】:{GIFT_REACTIONS[gift][self.name]}")
        self.change_affinity(GIFT_EFFECTS[gift][self.name])
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        pass

    def change_affinity(self, value):
        self.affinity += value
        print(f"【{self.name}】 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            ending = CHARACTER_ENDINGS[self.name]["textB"]

            print(f"\n{'=' * 100}")
            print(f"🎭 {self.name} - {"textB"} 结局")
            print(f"{'=' * 100}")

            print(f"{ending['prelude']}")
            sleep(3)
            print(f"{ending['confession']}")
            sleep(3)
            print(f"{ending['reaction']}")
            sleep(3)
            print(f"{ending['main']}")
            sleep(3)
            print(f"{ending['epilogue']}")
            sleep(3)
            print(f"{'=' * 100}")
            return True
        else:
            ending = CHARACTER_ENDINGS[self.name]["textA"]

            print(f"\n{'=' * 100}")
            print(f"🎭 {self.name} - {"textA"} 结局")
            print(f"{'=' * 100}")

            print(f"{ending['prelude']}")
            sleep(3)
            print(f"{ending['confession']}")
            sleep(3)
            print(f"{ending['reaction']}")
            sleep(3)
            print(f"{ending['main']}")
            sleep(3)
            print(f"{ending['epilogue']}")
            sleep(3)
            print(f"{'=' * 100}")
            return False


class Game:
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐")
        }
        self.current_target = None

    def start(self):
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 依次进入三个场景 如果三个都不选....
        if not self.scene_senpai():  # 学姐场景
            if not self.scene_xiaobai():  # 小白场景
                if not self.scene_jiejie():  # 姐姐场景
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
            print("在纵目睽睽下，你扬长而去。")
            return False

    def scene_xiaobai(self):
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print("\n你饶有兴趣地驻足，附身凑上前查看。")
            print("委屈巴巴的小白瞬间涨红了脸，紧张兮兮地指出了她的难处。你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("嘁，你轻蔑一瞥，心想着这么简单的题目，我姥姥都轻松秒杀，这傻逼九漏鱼脑子不用可以卖给别人，爷先行一步！")
            return False
        # TODO 两种选择 如果选择了1 则进入该位角色的故事线 并返回 True 如果选择了 2 则进入下一位角色的选择 并且返回False
        # 注意 除了判断外 你可以同时输出角色的反应
        # 比如在上一位角色的判断中 选择了1时 输出了print("\n你随手挑起一只笔，在纸上几笔勾勒出惊艳的图案，引得周围阵阵惊呼。")
        # 写法可以借鉴学姐线

    def scene_jiejie(self):
        print("\n【场景三：姐姐")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("『没什么，两个嵌套循环，最后组合输出而已。』\n")
            print("姐姐微微一笑，把手搭在了你的肩上。你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("你心想，傻逼吧我操，你妈我打代码的时候过来烦我，快滚把你我思路都毁了。")
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

            # TODO 完成输入不同选项时 进行的操作

            if choice == "1":
                self.current_target.talk()
                continue
            if choice == "2":
                if self.current_target.known:
                    print(f"请选择你要赠送给【{self.current_target.name}】的礼物")
                    for i,gifts in enumerate(GIFT_EFFECTS.keys(),1):
                        print(f"{i}:{gifts}")
                    choice=int(input())
                    gifts=list(GIFT_EFFECTS.keys())
                    self.current_target.give_gift(gifts[choice-1])
                else:
                    print(f"请先与【{self.current_target.name}】对话")
                continue
            if choice == "3":
                print(f"你和【{self.current_target.name}】的好感度是{self.current_target.affinity}")
                print(f"要和她告白吗?(1.是/2.否)")
                choice=int(input())
                if choice == 1:
                    self.current_target.check_ending()
                else:
                    print("虽然每天看见她你的心总是莫明其妙地扑通扑通跳，但还是没能鼓足勇气。\n\"再在相处一段时间吧\"，你这样对自己说到。\n故事结束…………")
                break
            if choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)

            else:
                print("无效输入，请重新选择。")



DIALOGUES = {

    "学姐": [
        {
            "text": "你是新来的吧？画画过吗？",
            "optionA": {
                "text": "其实我更擅长写代码，但我愿意试试画画。",
                "reaction": "学姐微微挑眉，嘴角露出一丝笑意：'有勇气尝试新事物，不错。'"
            },
            "optionB": {
                "text": "画画确实不太擅长，但我可以帮忙做其他事情。",
                "reaction": "学姐轻轻点头：'每个人都有自己的长处，能找到自己的位置就好。'"
            }
        },
        {
            "text": "这幅画……构图还算不错，你是随便画的吗？",
            "optionA": {
                "text": "嗯，我是随手画的，没想到能得到你的认可。",
                "reaction": "学姐轻轻点头：'天赋不错，但还需要更多练习。'"
            },
            "optionB": {
                "text": "不，我可是很认真的！因为你在看嘛。",
                "reaction": "学姐脸颊微红，别过脸去：'油嘴滑舌……不过画得确实还行。'"
            }
        },
        {
            "text": "加入社团后，不能三天打鱼两天晒网。",
            "optionA": {
                "text": "放心吧，我会认真对待。",
                "reaction": "学姐露出欣慰的笑容：'希望你能说到做到。'"
            },
            "optionB": {
                "text": "我会尽量安排时间来的，最近确实有点忙。",
                "reaction": "学姐理解地点点头：'学业重要，有空的时候过来就好。'"
            }
        },
        {
            "text": "我平时比较严格，你会不会觉得我很不好相处？",
            "optionA": {
                "text": "严格才好啊，我喜欢有目标的人。",
                "reaction": "学姐眼神柔和下来：'你能理解我的用心，我很高兴。'"
            },
            "optionB": {
                "text": "有时候是有点严厉，但我知道你是为我好。",
                "reaction": "学姐表情缓和：'你能这样想，说明你是个懂事的人。'"
            }
        },
        {
            "text": "今天的社团活动结束了，你要不要和我一起收拾？",
            "optionA": {
                "text": "当然可以，我来帮你。",
                "reaction": "学姐微笑着递给你画笔：'谢谢，有你帮忙快多了。'"
            },
            "optionB": {
                "text": "今天有点急事，下次一定帮忙。",
                "reaction": "学姐理解地笑了笑：'没关系，你先去忙吧。'"
            }
        }
    ],
    "小白": [
        {
            "text": "Python 里的 for 循环，我总是写不对……",
            "optionA": {
                "text": "来，我教你一个小技巧！",
                "reaction": "小白眼睛一亮，开心地凑近：'真的吗？太好了！'"
            },
            "optionB": {
                "text": "这个确实需要多练习，慢慢来就好。",
                "reaction": "小白认真地点点头：'嗯，我会继续努力的！'"
            }
        },
        {
            "text": "你看我借了好多书，会不会显得很傻？",
            "optionA": {
                "text": "不会啊，这说明你很爱学习，很可爱。",
                "reaction": "小白害羞地低下头：'真的吗？谢谢你这么说……'"
            },
            "optionB": {
                "text": "爱学习是好事，但也要注意休息。",
                "reaction": "小白感动地看着你：'你总是这么关心我……'"
            }
        },
        {
            "text": "写代码的时候，我总喜欢喝奶茶……你呢？",
            "optionA": {
                "text": "我也是！来，下次我请你一杯。",
                "reaction": "小白开心地拍手：'真的吗？那说定了哦！'"
            },
            "optionB": {
                "text": "我更喜欢喝茶，不过偶尔喝奶茶也不错。",
                "reaction": "小白好奇地问：'那你喜欢什么口味的？我们可以一起尝试。'"
            }
        },
        {
            "text": "你会不会觉得我太依赖你了？",
            "optionA": {
                "text": "依赖我也没关系，我喜欢这种感觉。",
                "reaction": "小白脸红了：'你这样说……我会更依赖你的。'"
            },
            "optionB": {
                "text": "互相帮助是应该的，不用想太多。",
                "reaction": "小白安心地笑了：'嗯，有你在我很放心。'"
            }
        },
        {
            "text": "要不要一起留在图书馆自习？我想有人陪。",
            "optionA": {
                "text": "好啊，我正好也要复习。",
                "reaction": "小白开心地整理书本：'太好了，有你在效率一定更高！'"
            },
            "optionB": {
                "text": "今天可能不行，不过明天可以陪你。",
                "reaction": "小白理解地点点头：'那说好了，明天一定要来哦！'"
            }
        }
    ],
    "姐姐": [
        {
            "text": "你也喜欢在咖啡店看书吗？",
            "optionA": {
                "text": "是啊，这里很安静，很适合思考。",
                "reaction": "姐姐温柔地笑了：'看来我们有很多共同点呢。'"
            },
            "optionB": {
                "text": "其实我是跟着你来的，想多了解你一些。",
                "reaction": "姐姐微微脸红：'你倒是很诚实呢……'"
            }
        },
        {
            "text": "研究生的生活，其实没有你想象的那么光鲜。",
            "optionA": {
                "text": "我能理解，压力一定很大吧。",
                "reaction": "姐姐感动地看着你：'谢谢你这么理解我……'"
            },
            "optionB": {
                "text": "不管怎样，你在我心里都很优秀。",
                "reaction": "姐姐眼神温柔：'有你在身边，感觉压力都小了很多。'"
            }
        },
        {
            "text": "你觉得，什么样的人最值得依靠？",
            "optionA": {
                "text": "稳重冷静的人。",
                "reaction": "姐姐若有所思：'看来你是个很理性的人呢。'"
            },
            "optionB": {
                "text": "真诚善良的人，就像你这样。",
                "reaction": "姐姐脸微微发红：'突然说这种话……'"
            }
        },
        {
            "text": "我常常一个人坐到很晚，你不会觉得孤单吗？",
            "optionA": {
                "text": "有时候孤单是种享受。",
                "reaction": "姐姐赞同地点头：'确实，独处的时候能想明白很多事情。'"
            },
            "optionB": {
                "text": "如果你愿意，我可以陪你。",
                "reaction": "姐姐眼神闪烁：'你这样说……我会当真的。'"
            }
        },
        {
            "text": "如果有一天，我遇到困难，你会帮我吗？",
            "optionA": {
                "text": "当然会，你不用一个人扛着。",
                "reaction": "姐姐眼眶微湿：'有你这句话，我就很安心了。'"
            },
            "optionB": {
                "text": "我会尽我所能帮助你。",
                "reaction": "姐姐温柔地笑了：'谢谢你，能认识你真好。'"
            }
        }
    ]
}

GIFT_EFFECTS = {
    # 通用 / 默认 值
    "鲜花": {"学姐": 10, "小白": 10, "姐姐": 15},
    "编程笔记": {"学姐": 5, "小白": 15, "姐姐": 15},
    "奶茶": {"学姐": 20, "小白": 20, "姐姐": 20},
    "奇怪的石头": {"学姐": -20, "小白": -20, "姐姐": -20},
    "精致的钢笔": {"学姐": 20, "小白": 10, "姐姐": 20},
    "可爱玩偶": {"学姐": 10, "小白": 20, "姐姐": 10},
    "夜宵外卖": {"学姐": 0, "小白": 5, "姐姐": -5}
}
GIFT_REACTIONS = {
    "鲜花": {
        "学姐": "哇！好漂亮的花！谢谢你这么用心，我真的很喜欢！",  # +10 热情大方
        "小白": "哼...花、花还不错啦...不过你别误会！",         # +10 害羞傲娇
        "姐姐": "这花真美，你的心意姐姐收到了，谢谢你。"         # +15 知心温柔
    },
    "编程笔记": {
        "学姐": "这个笔记很有用呢！我们一起学习进步吧！",         # +5 热情大方
        "小白": "这、这个对我很有帮助...谢谢你了。",            # +15 害羞傲娇
        "姐姐": "很实用的笔记呢，你真是个细心的孩子。"           # +15 知心温柔
    },
    "奶茶": {
        "学姐": "我最爱的奶茶！你太懂我了，一起喝吧！",          # +20 热情大方
        "小白": "奶、奶茶...你怎么知道我喜欢这个...",           # +20 害羞傲娇
        "姐姐": "暖暖的奶茶，暖暖的心意，谢谢你。"             # +20 知心温柔
    },
    "奇怪的石头": {
        "学姐": "什么鬼东西！",
        "小白": "傻逼吧，给我这个干嘛？",
        "姐姐": "………………"
    },
    "精致的钢笔": {
        "学姐": "这支钢笔太棒了！以后写代码都要用它！",          # +20 热情大方
        "小白": "笔...挺好看的，我会好好用的。",               # +10 害羞傲娇
        "姐姐": "很精致的钢笔呢，姐姐会好好珍惜的。"            # +20 知心温柔
    },
    "可爱玩偶": {
        "学姐": "好可爱！放在桌上陪我写代码吧！",               # +10 热情大方
        "小白": "玩、玩偶什么的...我才不喜欢呢！...不过谢谢",    # +20 害羞傲娇
        "姐姐": "很可爱的玩偶呢，看到它就会想起你。"            # +10 知心温柔
    },
    "夜宵外卖": {
        "学姐": "正好有点饿呢，谢谢你想着我！",                 # +0 热情大方
        "小白": "外、外卖？...其实我不太饿...不过还是谢谢",     # +5 害羞傲娇
        "姐姐": "晚上吃东西对身体不好呢，不过还是谢谢你的心意。" # -5 知心温柔
    }
}

CHARACTER_ENDINGS = {
    "学姐": {
        "textA": {
            "prelude": "社团展示日那天，你站在学姐的画作前，她正认真地给参观者讲解。当你走近时，她只是淡淡地瞥了你一眼，随即移开视线。",
            "confession": "学姐，这段时间在社团的相处让我明白了一件事...我喜欢你。不是因为你的画技有多好，而是你认真努力的样子深深吸引着我。",
            "reaction": "学姐停下手中的画笔，手指轻轻敲击画架，面无表情地转过身来，与你保持着明显的距离。",
            "main": "她双手抱胸，语气冰冷：'对不起，我不喜欢不努力的人，请离我远一点吧。'说着后退一步，'这段时间你的表现让我很失望。'",
            "epilogue": "她毫不犹豫地转身走向其他社员，连一个回眸都不曾给予。你看着她在画室忙碌的背影，明白你们之间始终隔着一道无法跨越的距离。窗外的阳光洒在她的画作上，却照不进你此刻冰冷的心情。"
        },
        "textB": {
            "prelude": "在社团年度画展的庆功宴上，学姐的作品获得了最高奖项。当你捧着花束向她祝贺时，她立即小跑着迎上来，眼中闪烁着兴奋的光芒。",
            "confession": "学姐！从第一次看你画画时，我就被你的专注打动了。你愿意...让这个总是给你添麻烦的学弟，成为能一直支持你的人吗？",
            "reaction": "学姐先是一愣，随后开心地扑进你怀里，双手紧紧环住你的脖子，把脸埋在你肩头。",
            "main": "她抬起头，双手捧住你的脸：'笨蛋，这一路走来多亏有你！'说着轻轻在你脸颊印下一吻，'没有你的支持和鼓励，我可能早就放弃了。'",
            "epilogue": "在众人的掌声和祝福中，她主动牵起你的手十指相扣，轻声在你耳边说：'以后我们要一起努力，永远不分开！' 窗外的樱花恰好飘落，她依偎在你怀里的身影温暖了整个春天。"
        }
    },
    "小白": {
        "textA": {
            "prelude": "图书馆的角落里，小白正在调试一个复杂的算法。你走过去想帮忙，她却立即合上笔记本，身体微微后仰与你拉开距离。",
            "confession": "小白...其实从教你编程开始，每次看到你认真的侧脸，我的心跳都会加速。我...我喜欢你，不是作为老师，而是作为一个男生。",
            "reaction": "小白猛地站起身，椅子发出刺耳的声响，她双手紧紧抱住胸前的书本，像是要筑起一道屏障。",
            "main": "她低着头，手指紧张地绞在一起：'这段时间谢谢你教我编程...但是...'她后退一步，'我们还是当朋友比较好。我觉得我们不太合适...'",
            "epilogue": "她几乎是逃跑般快步离开，连回头看一眼的勇气都没有。图书馆的灯光依旧明亮，却照不亮你此刻被拒绝的伤痛，只剩下冰冷的空气在你们之间流动。"
        },
        "textB": {
            "prelude": "在编程大赛夺冠后的庆功会上，小白喝了一点酒，兴奋地拉着你的手臂蹦蹦跳跳。回宿舍的路上，她主动挽着你的胳膊，月光温柔地洒在两人身上。",
            "confession": "小白，你知道吗？我最喜欢的不是教你写代码，而是看你学会时开心的笑容。所以...可以让我一直守护你的笑容吗？",
            "reaction": "小白惊喜地捂住嘴，随后开心地跳起来抱住你，把发烫的脸颊贴在你胸口。",
            "main": "你轻轻将她抵在墙边，她害羞地捶打你的胸口：'笨、笨蛋！谁允许你这样了...' 但双手却主动环上你的脖颈，'其实我也...最喜欢你了。'",
            "epilogue": "在温柔的月光下，她踮起脚尖在你唇上印下青涩的一吻。远处的路灯将相拥的身影拉得很长很长，空气中弥漫着甜蜜的气息，连晚风都带着幸福的味道。"
        }
    },
    "姐姐": {
        "textA": {
            "prelude": "咖啡馆的灯光昏黄，姐姐搅拌着已经冷掉的咖啡。当你坐下时，她下意识地将椅子往后挪了挪，保持着疏离的距离。",
            "confession": "姐姐，和你在一起的每个瞬间都让我感到安心。我不知道从什么时候开始，对你的感情已经超越了姐弟之情...我喜欢你。",
            "reaction": "姐姐的手指猛地收紧，咖啡杯发出清脆的碰撞声。她站起身，双手抱臂站在窗边，背对着你。",
            "main": "她转过身，脸上带着礼貌而疏远的微笑：'这段时间很感谢你的陪伴，但是对不起...'她轻轻摇头，'我以为我们只是姐弟关系。这样的感情对我来说太沉重了。'",
            "epilogue": "她拿起包快步离开，连外套都忘了拿。咖啡馆的门铃响起，冷风灌入室内，你独自坐在原地，感受着这份被拒绝的冰冷。桌上的咖啡已经完全凉透，就像你们之间的关系。"
        },
        "textB": {
            "prelude": "新年夜的钟声即将敲响，姐姐带你来到她最喜欢的天台。她自然地挽着你的手臂，指着远处开始绽放的烟花，脸上洋溢着幸福的笑容。",
            "confession": "姐姐，在咖啡店的每个午后，在天台看星星的每个夜晚，我都希望能永远这样陪在你身边。你愿意...给我这个机会吗？",
            "reaction": "姐姐的眼中闪着泪光，她张开双臂紧紧抱住你，把脸埋在你胸前，声音带着哽咽：'傻瓜...我等这句话好久了。'",
            "main": "她踮起脚尖，双手捧着你的脸，在漫天烟花下深情地吻住你：'这一年，谢谢你一直在我身边，让我不再孤单。'她的手指轻轻抚过你的脸颊，'以后每个重要时刻，我都要在你怀里看。'",
            "epilogue": "在绚烂的烟花下，她依偎在你怀中，你们的手指紧紧交缠。新年的钟声悠扬响起，五彩的烟花在夜空中绽放，她仰起头主动索吻，这一刻的温暖足以融化整个冬天的寒冷。"
        }
    }
}



if __name__ == "__main__":
    game = Game()
    game.start()