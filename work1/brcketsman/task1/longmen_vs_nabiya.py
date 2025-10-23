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
    print(" 【%s】HP: %s / %s"%(character_name,current_hp,max_hp))
    # 在这里写你的代码，用print()函数


# 任务二：掷骰子
def roll_dice(num_dice):
    n=0
    total_points=0
    while n<num_dice:
        n += 1
        count = random.randint(1, 6)
        total_points += count
    # 在这里写你的代码
    return total_points


# 任务三：选择长门的行动
def choose_nagato_action(nagato_hp, nabiya_hp):
    if nagato_hp<30:
        return 'defend'
    elif nabiya_hp<20:
        return 'special'
    else:
        return 'attack'


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice):
    """调用 roll_dice() 函数来计算伤害"""
    return roll_dice(num_dice)


# 任务五：计算防御值
def calculate_defense_value(num_dice):
    """调用 roll_dice() 函数来计算防御值"""
    return roll_dice(num_dice)


# 任务六：检查是否暴击 (BIG SEVEN)
def check_critical_hit(base_damage):
    """如果伤害 >= 18，返回 True，否则返回 False"""
    if base_damage>=18:
        return True
    return False

# 任务七：娜比娅的AI行动
def nabiya_ai_action(nabiya_hp):
    """如果娜比娅HP <= 40，返回 'defend'，否则返回 'attack'"""
    if nabiya_hp<=40:
        return 'defend'
    return 'attack'


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
    while True:
        print(f"\n======== 回合 {turn} ========")
        nabiya_defense_bonus = 0
        nagato_defense_bonus = 0
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)
        time.sleep(1.8)

        # 3. --- 长门的回合 ---
        print("\n>>> 长门的回合")
        action = choose_nagato_action(nagato_hp,nabiya_hp)
        
        # 用 if/elif/else 处理不同行动
        if action == 'attack':
            print('\n长门 发起了攻击!')
            b_damage = calculate_attack_damage(4)
            if check_critical_hit(b_damage):
                damage = max(0,b_damage*2-nagato_defense_bonus)
            else:
                damage = max(0,b_damage-nagato_defense_bonus)
            print('\n长门 造成了%s点伤害!'%damage)
            nabiya_hp = nabiya_hp - damage
        elif action == 'defend':
            print('\n长门 展现了威仪,进入了防御姿态!')
            defend = roll_dice(3)
            print('\n长门 获得了%s点威仪值!下次受到的伤害减少%s!'%(defend,defend))
            nagato_defense_bonus += defend
        else:
            print('\n长门 用出了「四万神的守护」!')
            spe = random.randint(1,2)
            if spe == 1:
                print('\n成功了,造成了30点伤害!')
                nabiya_hp -= 30
            else:
                print('\n“唔…失手了…”')
        
        # 4. 检查娜比娅是否被击败
        if nabiya_hp <= 0:
            print('\n娜比娅 被击败了!')
            break
        
        time.sleep(1.4)

        # 5. --- 娜比娅的回合 ---
        print("\n>>> 娜比娅的回合")
        action = nabiya_ai_action(nabiya_hp)
        if action == 'attack':
            print('\n娜比娅 发起了反击!')
            b_damage = calculate_attack_damage(4)
            if check_critical_hit(b_damage):
                damage = max(0,b_damage*2-nabiya_defense_bonus)
            else:
                damage = max(0,b_damage-nabiya_defense_bonus)
            print('\n娜比娅 造成了%s点伤害'%damage)
            nagato_hp = nagato_hp - damage
        else:
            print('\n娜比娅 进行了防御!')
            defend = roll_dice(3)
            print('\n娜比娅 获得了%s点防御值!下次受到的伤害减少%s!'%(defend,defend))
            nabiya_defense_bonus += defend

        
        # 6. 检查长门是否被击败
        if nagato_hp <= 0:
            print('\n长门被击败了...')
            break

        turn = turn + 1
        time.sleep(1.4)

