"""娜比娅偷吃事件：回合制战斗练习。"""

from __future__ import annotations

from typing import Literal
import random
import time

Action = Literal["attack", "defend", "special"]
NabiyaAction = Literal["attack", "defend"]
BattleResult = Literal["nagato", "nabiya", "draw"]

# 这些数值可以让战斗持续数个回合，同时让防御有用但不会抵消大多数普通攻击。
NAGATO_MAX_HP = 115
NABIYA_MAX_HP = 105
NAGATO_ATTACK_DICE = 3
NAGATO_DEFEND_DICE = 2
NABIYA_ATTACK_DICE = 3
NABIYA_DEFEND_DICE = 2
SPECIAL_ATTACK_DAMAGE = 24
SPECIAL_ATTACK_SUCCESS_RATE = 0.45
CRITICAL_HIT_THRESHOLD = 15

NAGATO_LOW_HP_THRESHOLD = 30
NABIYA_SPECIAL_HP_THRESHOLD = 20
NABIYA_DEFEND_HP_THRESHOLD = 40
MAX_BATTLE_TURNS = 50


def display_status(character_name: str, current_hp: int, max_hp: int) -> None:
    """输出角色状态，格式为：[角色名]HP: 当前生命值 / 最大生命值。"""
    # TODO：检查最大生命值是否合法，并使用 print() 输出角色状态。
    if max_hp <= 0:
        print("当前生命值小于0")
    print(f"[{character_name}]HP: {current_hp} / {max_hp}")


def roll_dice(num_dice: int) -> int:
    """投掷指定数量的六面骰子并返回点数总和。"""
    # TODO：先处理非法骰子数量，再用 while 循环调用 random.randint(1, 6)。
    if num_dice < 0:
        raise ValueError("骰子为负数")  # raise返回报错
    else:
        current_num = num_dice
        total_num = 0
        while current_num > 0:
            current_num -= 1
            a = random.randint(1, 6)
            total_num += a
    return total_num


def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> Action:
    """根据双方生命值选择长门的行动。"""
    # TODO：长门生命值低于 30 时防御，娜比娅生命值低于 20 时使用特殊攻击，
    # TODO：其余情况进行普通攻击；注意使用 if/elif/else 保持判断顺序。
    if nagato_hp < 30:
        return "defend"
    if nabiya_hp >= 20:
        return "attack"
    return "special"


def calculate_attack_damage(num_dice: int) -> int:
    """调用 roll_dice() 计算基础攻击伤害。"""
    # TODO：把骰子数量传给 roll_dice()，并返回它的结果。
    attack = roll_dice(num_dice)
    return attack


def calculate_defense_value(num_dice: int) -> int:
    """调用 roll_dice() 计算本回合的防御值。"""
    # TODO：把骰子数量传给 roll_dice()，并返回它的结果。
    defence = roll_dice(num_dice)
    return defence


def check_critical_hit(base_damage: int) -> bool:
    """判断基础伤害是否达到暴击阈值。"""
    # TODO：当基础伤害大于等于 CRITICAL_HIT_THRESHOLD 时返回 True。
    if base_damage >= CRITICAL_HIT_THRESHOLD:
        return True
    return False


def nabiya_ai_action(nabiya_hp: int) -> NabiyaAction:
    """根据娜比娅生命值选择她的行动。"""
    # TODO：娜比娅生命值小于等于 40 时防御，否则攻击。
    if nabiya_hp <= 40:
        return "defend"
    return "attack"


def calculate_final_damage(base_damage: int, defense_bonus: int) -> int:
    """用防御值抵消基础伤害，并返回不会小于零的最终伤害。"""
    # TODO：拒绝负数伤害或防御值，再计算 max(0, 基础伤害 - 防御值)。
    if base_damage < 0 or defense_bonus < 0:
        raise ValueError("数值不合理")
    true_damage = max(0, base_damage - defense_bonus)
    return true_damage


def apply_damage(current_hp: int, base_damage: int, defense_bonus: int = 0) -> int:
    """结算一次攻击并返回不会小于零的剩余生命值。"""
    # TODO：调用 calculate_final_damage()，再从当前生命值中扣除最终伤害。
    current_hp -= calculate_final_damage(base_damage, defense_bonus)
    current_hp = max(0, current_hp)
    return current_hp


def is_battle_over(nagato_hp: int, nabiya_hp: int) -> bool:
    """判断是否至少有一名角色的生命值归零。"""
    # TODO：只要任意一方 HP 小于等于 0，就返回 True。
    if nagato_hp <= 0 or nabiya_hp <= 0:
        return True
    return False


def get_battle_result(nagato_hp: int, nabiya_hp: int) -> BattleResult:
    """根据双方剩余生命值返回胜者或平局。"""
    # TODO：仅一方存活时返回对应结果；双方同时归零或都存活时返回 draw。
    if nagato_hp > 0 and nabiya_hp <= 0:
        return "nagato"
    elif nabiya_hp > 0 and nagato_hp <= 0:
        return "nabiya"
    return "draw"


def main_battle_loop(
    pause_seconds: float = 0.0,
    max_turns: int = MAX_BATTLE_TURNS,
) -> BattleResult:
    """运行完整战斗，并返回 nagato、nabiya 或 draw。"""
    # TODO：先检查 pause_seconds 和 max_turns 是否合理，不合理时抛出 ValueError。
    # TODO：初始化 nagato_hp、nabiya_hp、nagato_defense_bonus、
    # TODO：nabiya_defense_bonus，以及从 1 开始的 turn。
    #
    # TODO：战斗循环可以按照下面的结构开始：
    # while nagato_hp > 0 and nabiya_hp > 0 and turn <= max_turns:
    #     输出当前回合和双方状态。
    #
    # TODO：长门回合：
    # 1. 调用 choose_nagato_action(nagato_hp, nabiya_hp) 获取 action。
    # 2. action == "attack" 时，调用    ()；
    #    如果 check_critical_hit() 返回 True，就把基础伤害翻倍。
    # 3. action == "defend" 时，调用 calculate_defense_value()，
    #    把结果保存到 nagato_defense_bonus。
    # 4. action == "special" 时，用 random.random() 判断是否小于
    #    SPECIAL_ATTACK_SUCCESS_RATE；成功时使用 SPECIAL_ATTACK_DAMAGE。
    # 5. 造成伤害时统一调用 apply_damage()，并在攻击后清零对方的防御值。
    # 6. 长门行动后，如果 is_battle_over() 返回 True，使用 break 结束循环。
    #
    # TODO：娜比娅回合：
    # 1. 调用 nabiya_ai_action(nabiya_hp) 获取 enemy_action。
    # 2. attack 时调用 calculate_attack_damage()，再用 apply_damage()
    #    扣除长门生命值；defend 时保存娜比娅的防御值。
    # 3. 娜比娅的攻击或防御结束后，清零已经消耗的长门防御值。
    #
    # TODO：双方回合完成后让 turn 增加 1，并根据 pause_seconds 调用 time.sleep()。
    # TODO：循环结束后调用 get_battle_result()，输出中文结果并返回 result。
    if pause_seconds < 0:
        raise ValueError("时间不能为负数")
    if max_turns <= 0:
        raise ValueError("回合数不能为零")
    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1
    while nagato_hp > 0 and nabiya_hp > 0 and turn <= max_turns:
        base_damage = 0
        action = choose_nagato_action(nagato_hp, nabiya_hp)
        print(f"第 {turn} 回合")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("那比亚", nabiya_hp, NABIYA_MAX_HP)
        if action == "attack":
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)
            if check_critical_hit(base_damage):
                base_damage = base_damage * 2
        elif action == "defend":
            nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
        elif action == "special":
            if random.random() <= SPECIAL_ATTACK_SUCCESS_RATE:
                base_damage = SPECIAL_ATTACK_DAMAGE
        nabiya_hp = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
        nabiya_defense_bonus = 0
        base_damage = 0

        if is_battle_over(nagato_hp, nabiya_hp):
            break
        enemy_action = nabiya_ai_action(nabiya_hp)

        if enemy_action == "attack":
            base_damage = calculate_attack_damage(NABIYA_ATTACK_DICE)
            nagato_hp = apply_damage(nagato_hp, base_damage, nagato_defense_bonus)
        elif enemy_action == "defend":
            nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)

        nagato_defense_bonus = 0
        turn += 1
        time.sleep(pause_seconds)

    result = get_battle_result(nagato_hp, nabiya_hp)
    print(f"{result}")
    return result
