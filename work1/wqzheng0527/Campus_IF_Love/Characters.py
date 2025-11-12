import Dialogue
import save
class Character:
    def __init__(self, name, role, affinity=0):
        self.name = name
        self.role = role
        self.affinity = affinity
        save.current_data['character'] = name
        save.current_data['affinity'] = affinity

    #用Dialogue.py替代
    #def talk(self):
        #print(f"你正在和{self.name}对话...")


    def give_gift(self, gift):
        if self.affinity < 10 :
            print(f"{self.name}和你的好感度太低，不想接受你的礼物。")
            return
        else:
            print(f"你送给 {self.name} 一份 {gift}。")
            if gift in self.GIFT_EFFECTS:
                affinity_change = self.GIFT_EFFECTS.get(gift).get(self.name)
                self.change_affinity(affinity_change)
            else:
                print('你没找到合适的礼物')

        save.current_data['affinity'] = self.affinity
        save.quicksave()

    def change_affinity(self, value):
        self.affinity += value
        print(f"{self.name} 的好感度变化 {value} -> 当前好感度：{self.affinity}")
        save.current_data['affinity'] = self.affinity

    def check_ending(self):
        if self.affinity >= 100:
            print(f"恭喜！你和 {self.name} 的故事进入了结局线！")
            save.current_data['ending'] = 'True'
            save.quicksave()
            return True
        return False

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