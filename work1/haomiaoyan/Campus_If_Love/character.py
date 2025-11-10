import random
from typing import List,Dict
from story import DIALOGUES,GIFT_EFFECTS

class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self):
        print(f"你正在和{self.name}对话...")
        # 获取当前角色的对话库，如果不存在则为空列表
        dialogue_pool: List[Dict[str, str]] = DIALOGUES.get(self.name, [])
        if not dialogue_pool:
            print('她意味深长的看了你一眼，眼神里透露着不耐烦，（她似乎不想跟你说活） ')
            return

        entry = random.choice(dialogue_pool)
        print(self.name + "：" + entry["text"])
        print("A）" + entry["optionA"])
        print("B）" + entry["optionB"])
        choice = input("请选择 A 或 B（大小写均可）：").strip().upper()

        self.change_affinity(5)

        if choice == 'A':
            self.change_affinity(5)
            print(f'{self.name}微微一笑，似乎对你印象更好了!')

        elif choice == 'B':
            self.change_affinity(-5)
            print(f'在听清你说的话以后，{self.name}的表情变得冷淡起来')

        else:
            print('你沉默不语，气氛被你弄得很尬 (你的情商怎么这么低啊，galgame不适合你)')

        # TODO: 补充具体对话，对话内容可以从剧本里面截取 根据主人公的不同，使用不同的对话（你也可以根据好感度的不同/对话次数的不同 改变对话和选项）

    def give_gift(self, gift):
        print(f"你送给 {self.name} 一份 {gift}。")
        # TODO: 完成礼物好感度逻辑（送出不同礼物加不同的好感度） 并调用change_affinity（）函数 传入此次好感度变化的数值value
        # 礼物效果表（基于剧本）
        effects = GIFT_EFFECTS.get(gift)
        if not effects:
            print("她似乎不明白你在送什么？")
            return
        # 如果角色在表中有值就取，否则默认 0
        value = effects.get(self.name, effects.get("default", 0))
        # 应用变化
        self.change_affinity(value)

        if value > 0:
            print(f"{self.name}开心的接过礼物，眼神里闪烁着喜悦！！（她很喜欢）")

        elif value < 0:
            print(f"{self.name}有些不情愿的接过你的礼物 （噢！她可不喜欢这种礼物）")

        else:
            print("她很礼貌的接过你的礼物，不过你清楚她对这礼物并不感兴趣（下次长点心吧）")

    def change_affinity(self, value):
        self.affinity = max(0, self.affinity + value)
        sign = "+" if value >= 0 else ""
        print(f"{self.name} 的好感度变化 {sign}{value} -> 当前好感度：{self.affinity}")

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False