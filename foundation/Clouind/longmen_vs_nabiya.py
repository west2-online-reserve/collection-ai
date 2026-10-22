# longmen_vs_nabiya.py
# 请根据引导文档(README.md)的要求，完成下面的8个函数。

import random
import time
from itertools import count

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
    print(f"【{character_name}】HP: {current_hp} / {max_hp}")
    pass


# 任务二：掷骰子
def roll_dice(num_dice):
    """用while循环，模拟掷N个骰子，返回总点数,N即为num_dice,相当于一个骰子投掷N次"""
    total_points = 0
    count=0
    # 在这里写你的代码
    while count < num_dice:
        roll = random.randint(1, 6)
        total_points += roll
        count += 1                      
    return total_points

# 任务三：选择长门的行动
def choose_nagato_action(nagato_hp, nabiya_hp):
    """用if/elif/else，根据血量返回 'attack', 'defend', 或 'special'"""
    # 在这里写你的代码
    if nagato_hp < 30:
        return 'defend'
    elif nabiya_hp < 20:
        return 'special'
    else:
        return 'attack'


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice):
    """调用 roll_dice() 函数来计算伤害"""
    # 在这里写你的代码
    roll = roll_dice(num_dice)
    return roll


# 任务五：计算防御值
def calculate_defense_value(num_dice):
    """调用 roll_dice() 函数来计算防御值"""
    # 在这里写你的代码
    roll = roll_dice(num_dice)
    return roll



# 任务六：检查是否暴击 (BIG SEVEN)
def check_critical_hit(base_damage):
    """如果伤害 >= 18，返回 True，否则返回 False"""
    # 在这里写你的代码
    if base_damage >= 18:
        return True
    else:
        return False
    


# 任务七：娜比娅的AI行动
def nabiya_ai_action(nabiya_hp):
    """如果娜比娅HP <= 40，返回 'defend'，否则返回 'attack'"""
    # 在这里写你的代码
    if nabiya_hp <= 40:
        return 'defend'
    else:
        return 'attack'
    pass


# 任务八：核心战斗循环
def main_battle_loop():
    """
    这是最重要的部分！请根据下面的注释步骤来完成。
    
    适当的编写输出来说明战斗发生了什么，比如：
    print("长门：「感受BIG SEVEN的威力吧！」")
    print("💥「BIG SEVEN」触发！伤害翻倍！")
    """
    # 1. 初始化长门和娜比娅的HP，以及双方的防御值
    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1

    # 2. 编写 while 循环，在双方都存活时继续战斗
    # 注意，不需要你编写选择行动的代码，只需要编写行动后的逻辑即可
    # while ...

        # print(f"\n======== 回合 {turn} ========")
        # display_status("长门", nagato_hp, NAGATO_MAX_HP)
        # display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        # 3. --- 长门的回合 ---
        # print("\n>>> 长门的回合")
        # action = choose_nagato_action(...)
        
        # 用 if/elif/else 处理不同行动
        # if action == 'attack':
        #     ...
        # elif action == 'defend':
        #     ...
        # else: # special
        #     ...
        
        # 4. 检查娜比娅是否被击败
        # if nabiya_hp <= 0:
        #     ...
        
        # time.sleep(1)

        # 5. --- 娜比娅的回合 ---
        # print("\n>>> 娜比娅的回合")
        # (和长门回合逻辑类似)
        
        # 6. 检查长门是否被击败
        # if nagato_hp <= 0:
        #     ...

        # turn = turn + 1
        # time.sleep(1)
    
    # 在这里写你的代码
    turn = 0
    while nagato_hp > 0 and nabiya_hp > 0:
        turn += 1
        print(f"\n======== 回合 {turn} ========")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        """
        任务一函数:显示角色状态 display_status(character_name, current_hp, max_hp)
        任务二函数:掷骰子 roll_dice(num_dice)
        任务三函数:选择长门的行动 choose_nagato_action(nagato_hp, nabiya_hp)
        任务四函数:计算攻击伤害 calculate_attack_damage(num_dice)
        任务五函数:计算防御值 calculate_defense_value(num_dice)
        任务六函数:检查是否暴击 check_critical_hit(base_damage)
        任务七函数:娜比娅的AI行动 nabiya_ai_action(nabiya_hp)
        """
        # 长门的回合
        print("=*10长门的回合*10=\n")
        action = choose_nagato_action(nagato_hp, nabiya_hp)
        if action == 'attack':
            print("长门：「感受BIG SEVEN的威力吧！」")
            base_damage = calculate_attack_damage(4)  # 长门使用4个骰子进行攻击
            if check_critical_hit(base_damage):
                print("💥「BIG SEVEN」触发！伤害翻倍！")
                base_damage *= 2
            effective_damage = max(0, base_damage - nabiya_defense_bonus)
            nabiya_hp -= effective_damage
            print(f"在第{turn}回合，长门对娜比娅造成了 {effective_damage} 点伤害！")
        elif action == 'defend':
            nagato_defense_bonus = calculate_defense_value(3)  
            print(f"长门选择防御，防御值为 {nagato_defense_bonus}！")
        else:  # 四万神的守护(special)
            print("长门使用了四万神的守护！")
            nabiya_hp-= SPECIAL_ATTACK_DAMAGE
            print(f"在第{turn}回合，长门对娜比娅造成了 {SPECIAL_ATTACK_DAMAGE} 点特殊伤害！")

        if nabiya_hp <= 0:
            print("娜比娅被击败了！长门获胜！")
            break

        time.sleep(1)



        # 娜比娅的回合
        print("=*10娜比娅的回合*10=\n")
        action = nabiya_ai_action(nabiya_hp)
        if action=="attack":
            print("娜比娅选择攻击！")
            base_damage = calculate_attack_damage(4)  # 娜比娅使用4个骰子进行攻击
            if check_critical_hit(base_damage):
                print("💥「BIG SEVEN」触发！伤害翻倍！")
                base_damage *= 2
            effective_damage = max(0, base_damage - nagato_defense_bonus)
            nagato_hp -= effective_damage
            print(f"在第{turn}回合，娜比娅对长门造成了 {effective_damage} 点伤害！")
        elif action == 'defend':
                    nabiya_defense_bonus = calculate_defense_value(3)  
                    print(f"娜比娅选择防御，防御值为 {nabiya_defense_bonus}！")
        if nagato_hp <= 0:
            print("长门被击败了！娜比娅获胜！")
            break
        
