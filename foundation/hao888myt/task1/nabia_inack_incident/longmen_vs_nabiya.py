"""娜比娅偷吃事件：回合制战斗练习。"""

from __future__ import annotations

import random
import time
from typing import Literal

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
    if max_hp > 0 and max_hp >= current_hp:
        print(f"[{character_name}]HP: {current_hp} / {max_hp}")
    elif max_hp <= 0:
        raise ValueError(f"[{character_name}]最大HP: {max_hp} 不合法! 应大于0!")
    elif max_hp < current_hp:
        raise ValueError(
            f"[{character_name}]最大HP: {max_hp} 不合法! 应大于当前生命值: {current_hp}!"
        )


def roll_dice(num_dice: int) -> int:
    """投掷指定数量的六面骰子并返回点数总和。"""
    # TODO：先处理非法骰子数量，再用 while 循环调用 random.randint(1, 6)。
    if num_dice < 0:
        raise ValueError(f"当前骰子数量: {num_dice} 非法! 应大于等于0!")

    total_num: int = 0

    while num_dice > 0:
        total_num += random.randint(1, 6)
        num_dice -= 1

    return total_num


def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> Action:
    """根据双方生命值选择长门的行动。"""
    # TODO：长门生命值低于 30 时防御，娜比娅生命值低于 20 时使用特殊攻击，
    # TODO：其余情况进行普通攻击；注意使用 if/elif/else 保持判断顺序。
    if nagato_hp < 30:
        return "defend"
    elif nabiya_hp < 20:
        return "special"
    else:
        return "attack"


def calculate_attack_damage(num_dice: int) -> int:
    """调用 roll_dice() 计算基础攻击伤害。"""
    # TODO：把骰子数量传给 roll_dice()，并返回它的结果。
    return roll_dice(num_dice)


def calculate_defense_value(num_dice: int) -> int:
    """调用 roll_dice() 计算本回合的防御值。"""
    # TODO：把骰子数量传给 roll_dice()，并返回它的结果。
    return roll_dice(num_dice)


def check_critical_hit(base_damage: int) -> bool:
    """判断基础伤害是否达到暴击阈值。"""
    # TODO：当基础伤害大于等于 CRITICAL_HIT_THRESHOLD 时返回 True。
    return base_damage >= CRITICAL_HIT_THRESHOLD


def nabiya_ai_action(nabiya_hp: int) -> NabiyaAction:
    """根据娜比娅生命值选择她的行动。"""
    # TODO：娜比娅生命值小于等于 40 时防御，否则攻击。
    if nabiya_hp <= 40:
        return "defend"
    else:
        return "attack"


def calculate_final_damage(base_damage: int, defense_bonus: int) -> int:
    """用防御值抵消基础伤害，并返回不会小于零的最终伤害。"""
    # TODO：拒绝负数伤害或防御值，再计算 max(0, 基础伤害 - 防御值)。
    if base_damage < 0 or defense_bonus < 0:
        raise ValueError(f"当前伤害: {base_damage} 不合法! 应大于0!")
    elif defense_bonus < 0:
        raise ValueError(f"当前防御值: {defense_bonus} 不合法! 应大于0!")

    return max(0, base_damage - defense_bonus)


def apply_damage(current_hp: int, base_damage: int, defense_bonus: int = 0) -> int:
    """结算一次攻击并返回不会小于零的剩余生命值。"""
    # TODO：调用 calculate_final_damage()，再从当前生命值中扣除最终伤害。
    return max(current_hp - calculate_final_damage(base_damage, defense_bonus), 0)


def is_battle_over(nagato_hp: int, nabiya_hp: int) -> bool:
    """判断是否至少有一名角色的生命值归零。"""
    # TODO：只要任意一方 HP 小于等于 0，就返回 True。
    return nagato_hp <= 0 or nabiya_hp <= 0


def get_battle_result(nagato_hp: int, nabiya_hp: int) -> BattleResult:
    """根据双方剩余生命值返回胜者或平局。"""
    # TODO：仅一方存活时返回对应结果；双方同时归零或都存活时返回 draw。
    if (nagato_hp <= 0 and nabiya_hp <= 0) or (nagato_hp > 0 and nabiya_hp > 0):
        return "draw"
    elif nagato_hp > 0 and nabiya_hp <= 0:
        return "nagato"
    elif nagato_hp <= 0 and nabiya_hp > 0:
        return "nabiya"

    return "draw"


def main_battle_loop(
    pause_seconds: float = 0.0,
    max_turns: int = MAX_BATTLE_TURNS,
) -> BattleResult:
    if pause_seconds < 0:
        raise ValueError(f"当前pause_seconds: {pause_seconds} 不合法! 应大于等于0!")

    if max_turns <= 0:
        raise ValueError(f"当前pause_seconds: {max_turns} 不合法! 应大于0!")

    nagato_hp: int = NAGATO_MAX_HP
    nabiya_hp: int = NABIYA_MAX_HP

    nagato_defense_bonus: int = 0
    nabiya_defense_bonus: int = 0

    turn: int = 1

    while not is_battle_over(nagato_hp, nabiya_hp) and turn <= max_turns:
        print(f"\n======== 第 {turn} 回合 ========")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        print("\n>>> 长门的回合")

        match choose_nagato_action(nagato_hp, nabiya_hp):
            case "attack":
                base_damage: int = calculate_attack_damage(NAGATO_ATTACK_DICE)
                if check_critical_hit(base_damage):
                    base_damage *= 2

                nabiya_hp = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
                nabiya_defense_bonus = 0
                print(f"长门造成了伤害，娜比娅剩余 {nabiya_hp} 点生命值。")

            case "defend":
                nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
                print(f"长门进入防御姿态，获得 {nagato_defense_bonus} 点防御值。")

            case "special":
                base_damage: int = calculate_attack_damage(NAGATO_ATTACK_DICE)

                if random.random() < SPECIAL_ATTACK_SUCCESS_RATE:
                    base_damage = SPECIAL_ATTACK_DAMAGE
                    print(f"长门的特殊攻击成功，娜比娅剩余 {nabiya_hp} 点生命值。")
                else:
                    print("长门的特殊攻击失败了，伤害维持原样。")

                nabiya_hp = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
                nabiya_defense_bonus = 0

        if is_battle_over(nagato_hp, nabiya_hp):
            break

        print("\n>>> 娜比娅的回合")

        match nabiya_ai_action(nabiya_hp):
            case "attack":
                base_damage: int = calculate_attack_damage(NABIYA_ATTACK_DICE)
                nagato_hp = apply_damage(
                    nagato_hp,
                    base_damage,
                    nagato_defense_bonus,
                )
                nagato_defense_bonus = 0
                print(f"娜比娅发起攻击，长门剩余 {nagato_hp} 点生命值。")

            case "defend":
                nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)
                print(f"娜比娅进入防御姿态，获得 {nabiya_defense_bonus} 点防御值。")

        if is_battle_over(nagato_hp, nabiya_hp):
            break

        turn += 1
        time.sleep(pause_seconds)

    result = get_battle_result(nagato_hp, nabiya_hp)

    match result:
        case "draw":
            print("哦先生这似乎胜负未分")
        case "nabiya":
            print("恭喜Nabiya偷吃成功喵")
        case "nagato":
            print("恭喜Nagato保卫成功喵")

    return result
