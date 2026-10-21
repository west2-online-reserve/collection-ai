import json
import os

class Character:
    def __init__(self, name, role, affinity=0)->None:
        self.name = name
        self.role = role
        self.affinity = affinity
        self.count = 0   # 对话进度：每成功聊一次 +1，用来解锁下一段对话


    def save(self):
        data = {}
        for name, c in self.characters.items():
            if name == "None":
                continue
            data[name] = {"affinity": c.affinity, "count": c.count}
        with open("save.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("进度已保存。")

    def load(self):
        """读档：从 save.json 恢复三个角色的状态，成功返回 True"""
        if not os.path.exists("save.json"):
            print("没有找到存档。")
            return False
        try:
            with open("save.json", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("存档文件损坏，无法读取。")
            return False
        for name, d in data.items():
            if name in self.characters:
                self.characters[name].affinity = d["affinity"]
                self.characters[name].count = d["count"]
        print("读档成功！")
        return True

    def talk(self)->str:
        print(f"你正在和{self.name}对话...")
        if self.name == "学姐" :
            if self.affinity  > 0 and self.count == 0:
                print("学姐：你是新来的吧？画画过吗？")
                print("A: 其实我更擅长写代码，但我愿意试试画画。\nB: 画画？那种小孩子的东西我可没兴趣。")
                good = "A"
                self.count += 1
            elif self.affinity > 20 and self.count == 1:
                print("学姐：这幅画……构图还算不错，你是随便画的吗？")
                print("A: 嗯，我是随手画的，没想到能得到你的认可。\nB: 不，我可是很认真的！因为你在看嘛。")
                good = "B"
                self.count += 1
            elif self.affinity > 40 and self.count == 2:
                print("学姐：加入社团后，不能三天打鱼两天晒网。")
                print("A: 放心吧，我会认真对待。\nB: 社团只是玩玩而已吧？不要太严肃了。")
                good = "A"
                self.count += 1
            elif self.affinity > 60 and self.count == 3:
                print("学姐：我平时比较严格，你会不会觉得我很不好相处？")
                print("A: 严格才好啊，我喜欢有目标的人。\nB: 嗯……确实有点难相处。")
                good = "A"
                self.count += 1
            elif self.affinity > 80 and self.count == 4:
                print("学姐：今天的社团活动结束了，你要不要和我一起收拾？")
                print("A: 当然可以，我来帮你。\nB: 算了吧，我还有事要做。")
                good = "A"
                self.count += 1
            else:
                print("你还没有触发对话条件（好感度不足或没有更多对话了）。")
                return

        elif self.name == "小白":
            if self.affinity > 0 and self.count == 0:
                print("小白：Python 里的 for 循环，我总是写不对……")
                print("A: 来，我教你一个小技巧！\nB: 这种简单的东西，你都不会？")
                good = "A"
                self.count += 1
            elif self.affinity > 20 and self.count == 1:
                print("小白：你看我借了好多书，会不会显得很傻？")
                print("A: 不会啊，这说明你很爱学习，很可爱。\nB: 嗯……的确有点太贪心了。")
                good = "A"
                self.count += 1
            elif self.affinity > 40 and self.count == 2:
                print("小白：写代码的时候，我总喜欢喝奶茶……你呢？")
                print("A: 我也是！来，下次我请你一杯。\nB: 我只喝水，健康第一。")
                good = "A"
                self.count += 1
            elif self.affinity > 60 and self.count == 3:
                print("小白：你会不会觉得我太依赖你了？")
                print("A: 依赖我也没关系，我喜欢这种感觉。\nB: 嗯……是有点吧。")
                good = "A"
                self.count += 1
            elif self.affinity > 80 and self.count == 4:
                print("小白：要不要一起留在图书馆自习？我想有人陪。")
                print("A: 好啊，我正好也要复习。\nB: 算了，我还是回宿舍打游戏吧。")
                good = "A"
                self.count += 1
            else:
                print("你还没有触发对话条件（好感度不足或没有更多对话了）。")
                return

        elif self.name == "姐姐":
            if self.affinity > 0 and self.count == 0:
                print("姐姐：你也喜欢在咖啡店看书吗？")
                print("A: 是啊，这里很安静，很适合思考。\nB: 其实我只是随便找个地方坐。")
                good = "A"
                self.count += 1
            elif self.affinity > 20 and self.count == 1:
                print("姐姐：研究生的生活，其实没有你想象的那么光鲜。")
                print("A: 我能理解，压力一定很大吧。\nB: 那我还是不要考研了。")
                good = "A"
                self.count += 1
            elif self.affinity >40 and self.count == 2:
                print("姐姐：你觉得，什么样的人最值得依靠？")
                print("A: 稳重冷静的人。\nB: 长得好看的人。")
                good = "A"
                self.count += 1
            elif self.affinity >60 and self.count == 3:
                print("姐姐：我常常一个人坐到很晚，你不会觉得孤单吗？")
                print("A: 有时候孤单是种享受。\nB: 一个人太寂寞了，我受不了。")
                good = "A"
                self.count += 1
            elif self.affinity >80 and self.count == 4:
                print("姐姐：如果有一天，我遇到困难，你会帮我吗？")
                print("A: 当然会，你不用一个人扛着。\nB: 我大概帮不了你吧。")
                good = "A"
                self.count += 1
            else:
                print("你还没有触发对话条件（好感度不足或没有更多对话了）。")
                return

        else:
            print(f"{self.name}：……（对方似乎不太想说话）")
            return

        print("请输入你的选择：A/B")
        a = input().strip().upper()
        if a == good:
            self.change_affinity(5)
        elif a in ("A", "B"):
            self.change_affinity(-5)
        else:
            print("（你没有接话，气氛有点尴尬……）")

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
    def give_gift(self, gift)-> str:

        print("你可以赠送\n鲜花\n编程笔记\n奶茶\n奇怪的石头\n精致的钢笔\n可爱玩偶\n夜宵外卖\n请输入你要赠送的礼物：", end='')
        gift=input().strip()
        print(f"你送给 {self.name} 一份 {gift}。")
        if self.name=="学姐":
            if gift=="鲜花":
                if self.affinity<=50:
                    print("学姐：抱歉我不能接受你的鲜花")
                    self.change_affinity(-5)
                else:
                    print("学姐：谢谢你的鲜花")
                    self.change_affinity(10)
            elif gift=="编程笔记":
                print("学姐：谢谢你的笔记")
                self.change_affinity(5)
            elif gift=="奶茶":
                if self.affinity<20:
                    print("学姐：抱歉我不能接受你的奶茶")
                    self.change_affinity(0)
                else:
                    print("学姐：谢谢你的奶茶")
                    self.change_affinity(20)
            elif gift=="奇怪的石头":
                print("学姐(不失礼貌的微笑）：谢谢你的石头")
                self.change_affinity(-10)
            elif gift=="精致的钢笔":
                print("学姐：哇，哪来的这么好看的钢笔")
                self.change_affinity(20)
            elif gift=="可爱玩偶":
                print("学姐：哼，我才不是小孩子了……不过就收下吧。")
                self.change_affinity(10)
            elif gift=="夜宵外卖":
                print("学姐：谢谢你，我不爱吃外卖")
                self.change_affinity(0)
            else:
                print("学姐：这算什么礼物呀……")
                self.change_affinity(-5)
        elif self.name=="小白":
            if gift=="鲜花":
                print("小白：哇！鲜花诶……我、我才不会害羞呢！")
                self.change_affinity(10)
            elif gift=="编程笔记":
                print("小白：是你的笔记？！太好了，我可以偷偷学好多东西了！")
                self.change_affinity(15)
            elif gift=="奶茶":
                print("小白：你居然记得我爱喝奶茶！你是好人！")
                self.change_affinity(20)
            elif gift=="奇怪的石头":
                print("小白：……你为什么送我石头啊？")
                self.change_affinity(-10)
            elif gift=="精致的钢笔":
                print("小白：这么好看的笔，我要用它写出满分的代码！")
                self.change_affinity(10)
            elif gift=="可爱玩偶":
                print("小白：呜哇！好可爱！以后它就陪我一起写作业啦！")
                self.change_affinity(20)
            elif gift=="夜宵外卖":
                print("小白：深夜的炸鸡是学习的一部分！谢谢！")
                self.change_affinity(5)
            else:
                print("小白：诶？这是什么东西呀……")
                self.change_affinity(-5)
        elif self.name=="姐姐":
            if gift=="鲜花":
                print("姐姐：花很漂亮，谢谢你，是个有心思的孩子。")
                self.change_affinity(15)
            elif gift=="编程笔记":
                print("姐姐：整理得很清晰，你对待代码很认真呢。")
                self.change_affinity(15)
            elif gift=="奶茶":
                print("姐姐：正想喝点什么呢，你很贴心。")
                self.change_affinity(20)
            elif gift=="奇怪的石头":
                print("姐姐（忍着笑）：这就是你的幽默感吗？")
                self.change_affinity(-10)
            elif gift=="精致的钢笔":
                print("姐姐：我很喜欢这支笔，用它的日子也要努力哦。")
                self.change_affinity(20)
            elif gift=="可爱玩偶":
                print("姐姐：谢谢，虽然我早过了玩玩偶的年纪，但心意收到了。")
                self.change_affinity(10)
            elif gift=="夜宵外卖":
                print("姐姐：深夜吃外卖可不太健康呢，下次早点休息。")
                self.change_affinity(-5)
            else:
                print("姐姐：这个礼物……还挺特别的。")
                self.change_affinity(-5)
        else:
            print(f"{self.name}：……（对方不明所以地看着你手里的东西）")
            self.change_affinity(-5)

    def change_affinity(self, value)-> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self)->bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False
