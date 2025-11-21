# longmen_vs_nabiya.py
# 请根据引导文档(README.md)的要求，完成下面的8个函数。

import random
import time

# --- 战斗设定 (这些是预设好的值，不需要修改哦) ---
NAGATO_MAX_HP = 120
NABIYA_MAX_HP = 100
NAGATO_ATTACK_DICE = 4
NAGATO_DEFEND_DICE = 3
NABIYA_ATTACK_DICE = 4
NABIYA_DEFEND_DICE = 3
SPECIAL_ATTACK_DAMAGE = 30
CRITICAL_HIT_THRESHOLD = 18


# 任务一：显示角色状态
def display_status(character_name, current_hp, max_hp):
    """打印格式: 【角色名】HP: 当前血量 / 最大血量"""
    # 在这里写你的代码，用print()函数
    print(f"【{character_name}】HP： {current_hp} / {max_hp}")
    pass


# 任务二：掷骰子
def roll_dice(num_dice):
    """用while循环，模拟掷N个骰子，返回总点数"""
    total_points = 0
    count = 0
    # 在这里写你的代码
    while count<num_dice:
        temp=random.randint(1,6)
        total_points+=temp
        count+=1
    return total_points


# 任务三：选择长门的行动
def choose_nagato_action(nagato_hp, nabiya_hp):
    """用if/elif/else，根据血量返回 'attack', 'defend', 或 'special'"""
    # 在这里写你的代码
    if nagato_hp<30:
        return "defend"
    elif nabiya_hp<20:
        return "special"
    else:
        return "attack"
    pass


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice):
    """调用 roll_dice() 函数来计算伤害"""
    # 在这里写你的代码
    total_points=roll_dice(num_dice)
    if total_points>=18:
        return total_points*2
    else:
        return total_points
    pass


# 任务五：计算防御值
def calculate_defense_value(num_dice):
    """调用 roll_dice() 函数来计算防御值"""
    # 在这里写你的代码
    total_points=roll_dice(num_dice)
    return total_points
    pass


# 任务六：检查是否暴击 (BIG SEVEN)
def check_critical_hit(base_damage):
    """如果伤害 >= 18，返回 True，否则返回 False"""
    # 在这里写你的代码
    if base_damage>=18:
        return True
    return False
    pass


# 任务七：娜比娅的AI行动
def nabiya_ai_action(nabiya_hp):
    """如果娜比娅HP <= 40，返回 'defend'，否则返回 'attack'"""
    # 在这里写你的代码
    if nabiya_hp<=40:
        return "defend"
    return "attack"
    pass


# 任务八：核心战斗循环
def main_battle_loop():
    """
    这是最重要的部分！请根据下面的注释步骤来完成。
    
    适当的编写输出来说明战斗发生了什么，比如：
    print("长门：「感受BIG SEVEN的威力吧！」")
    print("💥「BIG SEVEN」触发！伤害翻倍！")
    """
    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1

    while nagato_hp>0 and nabiya_hp>0:
        flag=0
        flag1=0
        total_points1_1=0
        total_points1_2=0
        total_points2_1=0
        total_points2_2=0
        print(f"\n======== 回合 {turn} ========")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)
        print("\n>>> 长门的回合")
        print("首先选择长门的回合行动")
        action1 = choose_nagato_action(nagato_hp,nabiya_hp)
        if action1 == "attack":
            print("本回合长门选择进攻")
            total_points1_1=calculate_attack_damage(4)
            if check_critical_hit(total_points1_1):
                print("长门：「感受BIG SEVEN的威力吧！」")
                print("💥「BIG SEVEN」触发！伤害翻倍！")
        elif action1 == "defend":
            print("本回合长门选择防守")
            total_points1_2=calculate_defense_value(3)
        else:
            flag=1
            print("本回合长门选择赌运气！")
            random_float=random.random()
            if random_float>=0 and random_float<0.5:
                print("赌运气成功，召唤守护之力，对娜比娅造成固定的30点伤害")
                flag1=1
            else:
                print("赌运气失败，唔…失手了…，不造成任何伤害")
        print(f"\n>>> 娜比娅的回合")
        print("首先选择娜比娅的回合行动")
        action2 =nabiya_ai_action(nabiya_hp)
        if action2 == "attack":
            print("本回合娜比娅选择进攻")
            total_points2_1=calculate_attack_damage(4)
            if check_critical_hit(total_points2_1):
                print("娜比娅：「感受BIG SEVEN的威力吧！」")
                print("💥「BIG SEVEN」触发！伤害翻倍！")
        else:
            print("本回合娜比娅选择防守")
            total_points2_2=calculate_defense_value(3)
        if total_points1_1>0 and total_points2_1>0:
            nagato_hp-=total_points2_1
            if nabiya_hp<0:
                print("娜比娅已被击败，游戏结束")
                time.sleep(1)
                break
            nabiya_hp-=total_points1_1
            if nagato_hp<0:
                print("长门已被击败，游戏结束")
                time.sleep(1)
                break
        elif total_points1_1>0 and total_points2_2>0:
            if total_points2_2>=total_points1_1:
                pass
            else:
                nabiya_hp=nabiya_hp-total_points1_1+total_points2_2
                if nabiya_hp<0:
                    print("娜比娅已被击败，游戏结束")
                    time.sleep(1)
                    break
        elif total_points1_2>0 and total_points2_1>0:
            if total_points1_2>=total_points2_1:
                pass
            else:
                nagato_hp=nagato_hp-total_points2_1+total_points1_2
                if nagato_hp<0:
                    print("长门已被击败，游戏结束")
                    time.sleep(1)
                    break
        elif total_points1_2>0 and total_points2_2>0:
            print("无法决出胜负，游戏结束")
            break
        elif flag==1 and total_points2_2>0:
            if flag1==1:
                nabiya_hp-30+total_points2_2
                if nabiya_hp<0:
                    print("娜比娅已被击败，游戏结束")
                    time.sleep(1)
                    break
            else:
                pass
        else:
            print("娜比娅已被击败，游戏结束")
            time.sleep(1)
            break
        turn+=1
        time.sleep(1)
    pass
