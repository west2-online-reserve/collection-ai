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
    print("【%s】HP：%d/%d" % (character_name, current_hp, max_hp))
    # print(f"【{character_name}】HP：{current_hp}/{max_hp}")


# 任务二：掷骰子
def roll_dice(num_dice):
    """用while循环，模拟掷N个骰子，返回总点数"""
    total_points = 0
    count = 0
    while count < num_dice:
        total_points += random.randint(1, 6)
        count += 1
    return total_points


# 任务三：选择长门的行动
def choose_nagato_action(nagato_hp, nabiya_hp):
    """用if/elif/else，根据血量返回 'attack', 'defend', 或 'special'"""
    # 在这里写你的代码
    if nagato_hp < 30:
        return "defend"
    elif nabiya_hp < 20:
        return "special"
    else:
        return "attack"


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice):
    """调用 roll_dice() 函数来计算伤害"""
    # 在这里写你的代码
    return roll_dice(num_dice)


# 任务五：计算防御值
def calculate_defense_value(num_dice):
    """调用 roll_dice() 函数来计算防御值"""
    # 在这里写你的代码
    return roll_dice(num_dice)


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
        return "defend"
    else:
        return "attack"


# 任务八：核心战斗循环
def main_battle_loop():
    # 1. 初始化长门和娜比娅的HP，以及双方的防御值
    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1

    # 2. 编写 while 循环，在双方都存活时继续战斗
    while nagato_hp > 0 and nabiya_hp > 0:

        print(f"\n======== 回合 {turn} ========")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        # 3. --- 长门的回合 ---
        print("\n>>> 长门的回合")
        action = choose_nagato_action(nagato_hp, nabiya_hp)

        # 用 if/elif/else 处理不同行动
        if action == 'attack':
            print("这一回合，长门使用attack技能")
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)
            if check_critical_hit(base_damage):
                base_damage *= 2
            if base_damage > nabiya_defense_bonus:
                over_hp = base_damage - nabiya_defense_bonus
                nabiya_hp -= over_hp
                print(f"护盾被击碎！娜比娅仍收到{over_hp}点伤害！")
            else:
                nabiya_defense_bonus -= base_damage
                print(f"娜比娅护盾剩余{nabiya_defense_bonus}，未受伤！")

        elif action == 'defend':
            print("这一回合，长门使用defend技能")
            nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)

        else:
            print("这一回合，长门使用special技能")
            if random.random() < 0.5:
                nabiya_hp -= 30
            else:
                print("失手了")

        # 4. 检查娜比娅是否被击败
        if nabiya_hp <= 0:
            print("娜比娅成功被长门击败！")

        time.sleep(1)

        # 5. --- 娜比娅的回合 ---
        print("\n>>> 娜比娅的回合")
        # (和长门回合逻辑类似)

        # 6. 检查长门是否被击败
        if nagato_hp <= 0:
            print("长门被娜比娅击败！")

        action = nabiya_ai_action(nabiya_hp)

        if action == 'attack':
            print("这一回合，娜比娅使用attack技能")
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)
            if check_critical_hit(base_damage):
                base_damage *= 2
            if base_damage > nagato_defense_bonus:
                over_hp = base_damage - nagato_defense_bonus
                nagato_hp -= over_hp
                print(f"护盾被击碎！长门仍收到{over_hp}点伤害！")
            else:
                nagato_defense_bonus -= base_damage
                print(f"长门护盾剩余{nagato_defense_bonus}，未受伤！")
        elif action == 'defend':
            print("这一回合，娜比娅使用defend技能")
            nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)

        turn = turn + 1
        time.sleep(1)
        nagato_defense_bonus = 0
        nabiya_defense_bonus = 0