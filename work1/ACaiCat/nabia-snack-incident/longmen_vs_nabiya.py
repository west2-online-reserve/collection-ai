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
def display_status(character_name: str, current_hp: int, max_hp: int) -> None:
    """打印格式: 【角色名】HP: 当前血量 / 最大血量"""
    # 在这里写你的代码，用print()函数
    print(f"【{character_name}】HP: {current_hp} / {max_hp}")


# 任务二：掷骰子
def roll_dice(num_dice: int) -> int:
    """用while循环，模拟掷N个骰子，返回总点数"""
    total_points = 0
    count = 0
    while count < num_dice:
        total_points += random.randint(1, 6)
        count += 1
    return total_points


# 任务三：选择长门的行动
def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> str:
    """用if/elif/else，根据血量返回 'attack', 'defend', 或 'special'"""
    if nagato_hp < 30:
        return "defend"
    elif nabiya_hp < 20:
        return "special"
    else:
        return "attack"


# 任务四：计算攻击伤害
def calculate_attack_damage(num_dice: int) -> int:
    """调用 roll_dice() 函数来计算伤害"""
    return roll_dice(num_dice)


# 任务五：计算防御值
def calculate_defense_value(num_dice: int) -> int:
    """调用 roll_dice() 函数来计算防御值"""
    return roll_dice(num_dice)


# 任务六：检查是否暴击 (BIG SEVEN)
def check_critical_hit(base_damage: int) -> int:
    """如果伤害 >= 18，返回 True，否则返回 False"""
    return base_damage >= CRITICAL_HIT_THRESHOLD


# 任务七：娜比娅的AI行动
def nabiya_ai_action(nabiya_hp: int) -> str:
    """如果娜比娅HP <= 40，返回 'defend'，否则返回 'attack'"""
    return "defend" if nabiya_hp <= 40 else "attack"


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
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)
        print("\n>>> 长门的回合")

        match choose_nagato_action(nagato_hp, nabiya_hp):

            case 'attack':
                damage = calculate_attack_damage(NAGATO_ATTACK_DICE)

                if check_critical_hit(damage):
                    print("长门：「感受BIG SEVEN的威力吧！」")
                    print("💥「BIG SEVEN」触发！伤害翻倍！")
                    damage = damage * 2

                damage = max(0, damage - nabiya_defense_bonus)

                print(f"💥长门对娜比娅使用「炮击」，造成了{damage}点伤害!")
                nabiya_hp -= damage

            case 'defend':
                nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
                print(f"🛡️长门发动「威仪」进入防御姿态，获得了{nagato_defense_bonus}点威仪值!")

            case 'special':
                print("长门准备发动「四万神的守护」")
                if random.randint(0, 1) == 1:
                    print(f"💥守护之力召唤成功，造成了30点伤害!")
                    damage = max(0, SPECIAL_ATTACK_DAMAGE - nabiya_defense_bonus)
                    nabiya_hp -= damage

                else:
                    print(f"什么也没有发生...")
                    print("长门：唔…失手了…")

        if nabiya_hp <= 0:
            print("娜比娅寄了，长门获胜~")
            break

        time.sleep(1)

        print("\n>>> 娜比娅的回合")

        match nabiya_ai_action(nabiya_hp):
            case 'attack':
                damage = calculate_attack_damage(NABIYA_ATTACK_DICE)
                damage = max(0, damage - nagato_defense_bonus)

                print(f"💥娜比娅对长门造成了{damage}点伤害!")
                nagato_hp -= damage

            case 'defend':
                nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)
                print(f"🛡️娜比娅进入防御姿态，获得了{nabiya_defense_bonus}点防御值!")

        if nagato_hp <= 0:
            print("长门寄了，娜比娅获胜~")
            break

        nabiya_defense_bonus = 0
        nagato_defense_bonus = 0

        turn = turn + 1
        time.sleep(1)
