import sys
from typing import Dict, List, Optional
from story import DIALOGUES, GIFT_EFFECTS
from manage import save_progress, load_progress

class Character:
    name: str
    role: str
    affinity: int

    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name = name
        self.role = role
        self.affinity = affinity

    def talk(self) -> None:
        print(f"你正在和{self.name}对话...")
        l_senpai: List[Dict[str, str]] = DIALOGUES.get(self.name, [])
        for dialogue in l_senpai:
            print(f"{self.name}：『{dialogue['text']}』")
            print("1.", dialogue["optionA"])
            print("2.", dialogue["optionB"])
            choice = input("请选择：")
            if choice == "1":
                print(f"\n你选择了：{dialogue['optionA']}")
                self.change_affinity(10)
            elif choice == "2":
                print(f"\n你选择了：{dialogue['optionB']}")
        self.change_affinity(5)

    def give_gift(self, gift: str) -> None:
        print(f"你送给 {self.name} 一份 {gift}。")
        effects: Dict[str, int] = GIFT_EFFECTS.get(gift, {})
        value = effects.get(self.name, effects.get("default", 0))
        self.change_affinity(int(value))

    def change_affinity(self, value: int) -> None:
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            return True
        return False

class Game:
    characters: Dict[str, Character]
    current_target: Optional[Character]

    def __init__(self) -> None:
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
        }
        self.current_target = None

    def load_archive(self) -> None:
        data = load_progress()
        if data:
            # 恢复角色好感度
            char_affinity = data.get("characters", {})
            for name, affinity in char_affinity.items():
                if name in self.characters:
                    self.characters[name].affinity = affinity
            # 恢复当前目标角色
            target_name = data.get("current_target")
            if target_name and target_name in self.characters:
                self.current_target = self.characters[target_name]
            print("存档已加载！")
        else:
            print("没有找到存档，开始新游戏。")

    def start(self) -> None:
        print("========== 游戏开始：校园 if·恋 ==========")
        print("你是一名刚刚踏入大学校园的新生。")
        print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

        # 启动时询问是否读档
        load_choice = input("是否读取存档？(y/n): ")
        if load_choice.lower() == "y":
            self.load_archive()

        if not self.scene_senpai():
            if not self.scene_xiaobai():
                if not self.scene_jiejie():
                    print("\n啥，眼前三妹子都不要？？死现充别玩galgame")

    def scene_senpai(self) -> bool:
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

    def scene_xiaobai(self) -> bool:
        print("\n【场景二：小白】")
        print("你走进图书馆，发现小白正在奋笔疾书，却被一道算法题难住了。")
        print("小白：『呜呜……这题到底该怎么写呀？』")

        choice = input("1. 主动帮她解题\n2. 敷衍几句，转身离开\n请选择：")
        if choice == "1":
            print("\n你耐心地为小白讲解算法思路，她的眼睛顿时亮了起来。")
            print("小白：『哇，谢谢你！你真是太厉害了！』你进入【小白线】！")
            self.current_target = self.characters["小白"]
            self.story_loop()
            return True
        else:
            print("小白看着你离去的背影，失落地叹了口气。")
            return False

    def scene_jiejie(self) -> bool:
        print("\n【场景三：姐姐】")
        print("你偶然在校外的咖啡店敲代码,一位看起来成熟知性的姐姐似乎对你感兴趣，缓缓朝你走了过来...")
        print("姐姐：『你的代码思路很有趣呢，能给我讲讲你的实现方法吗？』")

        choice = input("1. 缓缓低眉，毫不在意的开始解释\n2. 头也不抬，保持敲代码的状态\n请选择：")
        if choice == "1":
            print("\n你详细地向姐姐解释了你的代码实现，她露出了欣赏的笑容。")
            print("姐姐：『真是个有才华的年轻人呢！』你进入【姐姐线】！")
            self.current_target = self.characters["姐姐"]
            self.story_loop()
            return True
        else:
            print("姐姐看着你专注敲代码的样子，微微一笑，转身离开了咖啡店。")
            return False

    def story_loop(self) -> None:
        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")
            print("5. 存档")
            print("6. 读档")

            choice = input("请输入选项：")

            if choice == "1":
                if self.current_target:
                    self.current_target.talk()
                else:
                    print("当前没有目标角色。")
            elif choice == "2":
                print("你有什么礼物想送给她？")
                print("1. 鲜花")
                print("2. 编程笔记")
                print("3. 奶茶")
                print("4. 奇怪的石头")
                print("5. 精致的钢笔")
                print("6. 可爱玩偶")
                print("7. 夜宵外卖")

                gift_choice = input("请选择礼物：")
                gifts: Dict[str, str] = {
                    "1": "鲜花",
                    "2": "编程笔记",
                    "3": "奶茶",
                    "4": "奇怪的石头",
                    "5": "精致的钢笔",
                    "6": "可爱玩偶",
                    "7": "夜宵外卖",
                }

                gift = gifts.get(gift_choice)
                if gift:
                    if self.current_target:
                        self.current_target.give_gift(gift)
                else:
                    print("无效的礼物选择。")
            elif choice == "3":
                if self.current_target:
                    print(f"{self.current_target.name} 的当前好感度：{self.current_target.affinity}")
                else:
                    print("当前没有目标角色。")
            elif choice == "4":
                print("你选择离开，游戏结束。")
                sys.exit(0)
            elif choice == "5":
                data = {
                    "characters": {name: char.affinity for name, char in self.characters.items()},
                    "current_target": self.current_target.name if self.current_target else None
                }
                save_progress(data)
                print("游戏进度已保存！")
            elif choice == "6":
                self.load_archive()
            else:
                print("无效输入，请重新选择。")

            if self.current_target and self.current_target.check_ending():
                break

if __name__ == "__main__":
    game = Game()
    game.start()
