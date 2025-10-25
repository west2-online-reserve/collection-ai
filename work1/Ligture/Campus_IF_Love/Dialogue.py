from Characters import Character
import save
#序号100及以后为非主线
DIALOGUES = {
    '学姐': {
        '1.1': {
            'text': '学姐:这幅画……构图还算不错，你是随便画的吗？',
            'A': {'text': '嗯，我是随手画的，没想到能得到你的认可。', 'next': '1.2', 'affinity': 2},
            'B': {'text': '不，我可是很认真的！因为你在看嘛。', 'next': '1.3', 'affinity': 5},
        },
        '1.2': {
            'text': '学姐:那你的画功很厉害啊，考虑加入我们美术社吗？',
            'A': {'text': '谢谢学姐，但我更喜欢编程(死宅)。', 'next': '2.1', 'affinity': -5},
            'B': {'text': '可以吗学姐，我很想加入美术社。', 'next': '1.4', 'affinity': 5},
        },
        '1.3': {
            'text': '学姐:看在你这么认真的份上，我可以考虑让你加入我们美术社，你感兴趣吗？',
            'A': {'text': '你算什么东西这么说话。', 'next': 0, 'affinity': -20,'narration':'学姐愤怒地离开了。'},
            'B': {'text': '当然了，感谢学姐给我这个机会。', 'next': '1.4', 'affinity': 3},
        },
        '1.4': {
            'text': '学姐:加入社团后，不能三天打鱼两天晒网。',
            'A': {'text': '放心吧，我会认真对待。', 'next': '2.1', 'affinity': 5,'narration':'学姐对你的态度很满意，同意你加入美术社，好感度提升！','do':"save.current_data['join_club']=True"},
            'B': {'text': '社团只是玩玩而已吧？不要太严肃了。', 'next': 0, 'affinity': -5,'narration':'学姐觉得你不够认真，让你好好想想再来找她。'},
        },
        '2.1': {
            'condition': False,
            'text': '不会写',
            'A': {'text': '', 'next': 0, 'affinity': 0,'narration':''},
            'B': {'text': '', 'next': 0, 'affinity': 0,'narration':''},
        },
        '100': {
            #加入社团才能触发
            'condition': lambda: save.current_data.get('join_club', False) is True,
            'text': '学姐:今天的社团活动结束了，你要不要和我一起收拾？',
            'A': {'text': '当然可以，我来帮你。', 'next': 0, 'affinity': 2,'narration':'你们一起收拾完教室，学姐对你印象更好了。'},
            'B': {'text': '算了吧，我还有事要做。', 'next': 0, 'affinity': -2,'narration':'学姐觉得你不够合群，有点失望。'},
        },
        '101': {
            #1.1对话中选择了B且好感度达到30
            'condition': lambda: save.current_data['choice'].get('1.1','A') == 'B' and save.current_data['affinity'] >= 30,
            'text': '学姐:今天有个艺术展，你想和我一起去吗？',
            'A': {'text': '好啊，我很喜欢艺术展览。', 'next': 0, 'affinity': 3,'narration':'你们在艺术展览中度过了愉快的时光，学姐对你更有好感了。'},
            'B': {'text': '不了，我对艺术不太感兴趣。', 'next': 0, 'affinity': -3,'narration':'学姐觉得你不够重视她的兴趣，有点失望。'},
        },
        #剧情来自Pycharm智能补全
        '102': {
            'text': '学姐:我最近遇到一个问题，能不能请你帮我个忙？',
            'A': {'text': '当然可以，学姐需要我做什么？', 'next': 0, 'affinity': 5,'narration':'你帮学姐完成了任务，她对你的好感度大幅提升。'},
            'B': {'text': '抱歉，我最近也很忙，可能帮不上忙。', 'next': 0, 'affinity': -5,'narration':'学姐觉得你不够可靠，有点失望。'},
        },
    },

    '小白': {
            '1.1': {
                'text': '小白:Python 里的 for 循环，我总是写不对……',
                'A': {'text': '来，我教你一个小技巧！', 'next': '1.2', 'affinity': 3},
                'B': {'text': '这种简单的东西，你都不会？', 'next': '1.3', 'affinity': -3},
            },
            '1.2': {
                'text': '小白:谢谢你！你会不会觉得我很笨？',
                'A': {'text': '不会啊，大家都是从不会到会的。', 'next': '1.4', 'affinity': 3},
                'B': {'text': '有点，不过多练习就好了。', 'next': '1.3', 'affinity': -2},
            },
            '1.3': {
                'text': '小白:呜……我会努力的，不给你添麻烦。',
                'A': {'text': '加油，我相信你能学会。', 'next': '1.4', 'affinity': 2},
                'B': {'text': '别太依赖我了。', 'next': 0, 'affinity': -3, 'narration': '小白有些失落，与你的距离拉远了。'},
            },
            '1.4': {
                'text': '小白:你看我借了好多书，会不会显得很傻？',
                'A': {'text': '不会啊，这说明你很爱学习，很可爱。', 'next': '2.1', 'affinity': 5, 'narration': '小白开心地笑了，好感度提升！'},
                'B': {'text': '嗯……的确有点太贪心了。', 'next': 0, 'affinity': -2, 'narration': '小白有点不好意思，气氛有些尴尬。'},
            },
            '2.1': {
                'condition': False,
                'text': '',
                'A': {'text': '', 'next': 0, 'affinity': 0, 'narration': ''},
                'B': {'text': '', 'next': 0, 'affinity': 0, 'narration': ''},
            },
            '100': {
                'condition': lambda: save.current_data.get('affinity', 0) >= 20,
                'text': '小白:要不要一起留在图书馆自习？我想有人陪。',
                'A': {'text': '好啊，我正好也要复习。', 'next': 0, 'affinity': 3, 'narration': '你们一起学习，关系更亲近了。'},
                'B': {'text': '算了，我还是回宿舍打游戏吧。', 'next': 0, 'affinity': -2, 'narration': '小白有点失望，一个人留下了。'},

            },
            #结局线对话
            '101': {
                'condition': lambda: save.current_data.get('ending',False) == True,
                'text': '小白:今天去电影院吗？',
                'A': {'text': '拉起她的手，"走。"', 'next': 0, 'affinity': 5, 'narration': '小白感激地看着你，好感度大幅提升。'},
                'B': {'text': '算了吧我今天有点事。', 'next': 0, 'affinity': -3, 'narration': '小白有些失落，觉得你不够支持她。'},
            },
    },
    '姐姐': {
        '1.1': {
            'text': '姐姐:你也喜欢在咖啡店看书吗？',
            'A': {'text': '是啊，这里很安静，很适合思考。', 'next': '1.2', 'affinity': 3},
            'B': {'text': '其实我只是随便找个地方坐。', 'next': '1.3', 'affinity': -2},
        },
        '1.2': {
            'text': '姐姐:研究生的生活，其实没有你想象的那么光鲜。',
            'A': {'text': '我能理解，压力一定很大吧。', 'next': '1.4', 'affinity': 3},
            'B': {'text': '那我还是不要考研了。', 'next': '1.3', 'affinity': -2},
        },
        '1.3': {
            'text': '姐姐:每个人都有自己的选择，开心就好。',
            'A': {'text': '谢谢姐姐的理解。', 'next': '1.4', 'affinity': 2},
            'B': {'text': '其实我还是很迷茫。', 'next': 0, 'affinity': -2, 'narration': '姐姐轻轻叹了口气，似乎有些担心你。'},
        },
        '1.4': {
            'text': '姐姐:你觉得，什么样的人最值得依靠？',
            'A': {'text': '稳重冷静的人。', 'next': '2.1', 'affinity': 3, 'narration': '姐姐微笑点头，对你的看法很认可。'},
            'B': {'text': '长得好看的人。', 'next': 0, 'affinity': -3, 'narration': '姐姐有些无语，觉得你不太靠谱。'},
        },
        '2.1': {
            'condition': False,
            'text': '',
            'A': {'text': '', 'next': 0, 'affinity': 0, 'narration': ''},
            'B': {'text': '', 'next': 0, 'affinity': 0, 'narration': ''},
        },
        '100': {
            'condition': lambda: save.current_data.get('affinity', 0) >= 20,
            'text': '姐姐:我常常一个人坐到很晚，你不会觉得孤单吗？',
            'A': {'text': '有时候孤单是种享受。', 'next': 0, 'affinity': 3, 'narration': '姐姐觉得你很懂她，关系更近了一步。'},
            'B': {'text': '一个人太寂寞了，我受不了。', 'next': 0, 'affinity': -2, 'narration': '姐姐有些失落，没再多说什么。'},
        },
        '101': {
            'condition': lambda: save.current_data['choice'].get('1.4','A') == 'A' and save.current_data['affinity'] >= 30,
            'text': '姐姐:如果有一天，我遇到困难，你会帮我吗？',
            'A': {'text': '当然会，你不用一个人扛着。', 'next': 0, 'affinity': 5, 'narration': '姐姐感激地看着你，好感度大幅提升。'},
            'B': {'text': '我大概帮不了你吧。', 'next': 0, 'affinity': -3, 'narration': '姐姐有些失望，觉得你不够可靠。'},
        },
    },
}


class Dialogue:
    def __init__(self, character:Character,node):
        self.character = character
        self.node = node

    def start_node(self, node:str):
        if DIALOGUES.get(self.character.name).get(node) is None:
            return False
        current_node = DIALOGUES.get(self.character.name).get(node)
        #不满足触发条件跳过
        if float(node) < 100: #只保存主线
            save.current_data['story_progress'] = node
        save.quicksave()
        if current_node.get('condition','True') is False:
            return False
        print(f"\n你正在和{self.character.name}对话...")
        print(current_node["text"])
        print(f'A选项: {current_node["A"]["text"]}')
        print(f'B选项: {current_node["B"]["text"]}')
        choice = input("请选择A或B：")

        if choice in ['A', 'B']:
            save.current_data['choice'][node] = choice
            option = current_node[choice]
            if 'affinity' in option:
                self.character.change_affinity(option['affinity'])
            if 'narration' in option:
                print(option['narration'])
            if 'do' in option:
                exec(option['do'])
            if option.get('next', 0) != 0:
                self.start_node(option['next'])
            return True
        else:
            print("无效选择，请选择A或B。")
            self.start_node(node)
            return False

    def start_dialogue(self,node='1.1'):

        return self.start_node(node)






