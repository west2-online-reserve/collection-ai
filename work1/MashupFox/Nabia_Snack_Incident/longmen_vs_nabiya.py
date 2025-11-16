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
    print(f"【{character_name}】HP: {current_hp} / {max_hp}")
    pass


# 任务二：掷骰子
def roll_dice(num_dice):
    """用while循环，模拟掷N个骰子，返回总点数"""
    total_points = 0
    count = 0
    # 在这里写你的代码
    while count < num_dice:
        total_points += random.randint(1, 6)
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
    pass


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice):
    """调用 roll_dice() 函数来计算伤害"""
    # 在这里写你的代码
    return roll_dice(num_dice)
    pass


# 任务五：计算防御值
def calculate_defense_value(num_dice):
    """调用 roll_dice() 函数来计算防御值"""
    # 在这里写你的代码
    return roll_dice(num_dice)
    pass


# 任务六：检查是否暴击 (BIG SEVEN)
def check_critical_hit(base_damage):
    """如果伤害 >= 18，返回 True，否则返回 False"""
    # 在这里写你的代码
    return base_damage >= CRITICAL_HIT_THRESHOLD
    pass


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
    
    while nagato_hp > 0 and nabiya_hp > 0:
        print(f"\n======== 回合 {turn} ========")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        # 3. --- 长门的回合 ---
        print("\n>>> 长门的回合")
        action = choose_nagato_action(nagato_hp, nabiya_hp)
        
        # 用 if/elif/else 处理不同行动
        if action == 'attack':
            # 炮击攻击
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)
            if check_critical_hit(base_damage):
                final_damage = base_damage * 2
                print(f"长门发动炮击！触发「BIG SEVEN」暴击！造成{final_damage}点伤害！")
            else:
                final_damage = base_damage
                print(f"长门发动炮击！造成{final_damage}点伤害！")
            
            # 应用娜比娅的防御
            actual_damage = max(0, final_damage - nabiya_defense_bonus)
            if nabiya_defense_bonus > 0:
                print(f"娜比娅的防御抵消了{nabiya_defense_bonus}点伤害！")
            
            nabiya_hp -= actual_damage
            nabiya_defense_bonus = 0  # 防御值使用后重置
            
        elif action == 'defend':
            # 防御行动
            nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
            print(f"长门展现威仪，进入防御姿态！获得{nagato_defense_bonus}点防御！")
            
        else: # special
            # 特殊攻击
            if random.random() < 0.5:  # 50%成功率
                actual_damage = max(0, SPECIAL_ATTACK_DAMAGE - nabiya_defense_bonus)
                if nabiya_defense_bonus > 0:
                    print(f"长门发动「四万神的守护」！娜比娅的防御抵消了{nabiya_defense_bonus}点伤害！")
                print(f"长门发动「四万神的守护」！造成{actual_damage}点伤害！")
                nabiya_hp -= actual_damage
            else:
                print("长门发动「四万神的守护」！唔…失手了…")
            
            nabiya_defense_bonus = 0  # 防御值使用后重置
        
        # 4. 检查娜比娅是否被击败
        if nabiya_hp <= 0:
            nabiya_hp = 0
            print("娜比娅被击败了！")
            break
        
        time.sleep(1)

        # 5. --- 娜比娅的回合 ---
        print("\n>>> 娜比娅的回合")
        nabiya_action = nabiya_ai_action(nabiya_hp)
        
        if nabiya_action == 'attack':
            # 娜比娅攻击
            damage = random.randint(15, 25)
            actual_damage = max(0, damage - nagato_defense_bonus)
            
            if nagato_defense_bonus > 0:
                print(f"娜比娅发动攻击！造成{damage}点伤害，长门的防御抵消了{nagato_defense_bonus}点，实际造成{actual_damage}点伤害！")
            else:
                print(f"娜比娅发动攻击！造成{actual_damage}点伤害！")
            
            nagato_hp -= actual_damage
            nagato_defense_bonus = 0  # 防御值使用后重置
            
        elif nabiya_action == 'defend':
            # 娜比娅防御
            nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)
            print(f"娜比娅进入防御姿态！获得{nabiya_defense_bonus}点防御！")
        
        # 6. 检查长门是否被击败
        if nagato_hp <= 0:
            nagato_hp = 0
            print("长门被击败了！")
            break

        turn = turn + 1
        time.sleep(1)
    
    # 战斗结束，显示结果
    print("\n=== 战斗结束 ===")
    display_status("长门", nagato_hp, NAGATO_MAX_HP)
    display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)
    
    if nagato_hp > 0:
        print("长门大人成功赶跑了偷吃军粮的娜比娅！港区的和平得到了维护！")
        return True
    else:
        print("娜比娅成功逃脱了！军粮保卫战失败...")
        return False
    pass