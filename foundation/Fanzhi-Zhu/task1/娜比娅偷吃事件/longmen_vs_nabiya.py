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
    """1.输出角色状态，格式为：[角色名]HP: 当前生命值 / 最大生命值。"""
    # 最大生命值必须为正数，当前生命值不能为负数。
    if max_hp <= 0:
        raise ValueError("最大生命值必须大于 0")
    if current_hp < 0:
        raise ValueError("当前生命值不能为负数")

    print(f"[{character_name}]HP: {current_hp} / {max_hp}")


def roll_dice(num_dice: int) -> int:
    """2.投掷指定数量的六面骰子并返回点数总和。"""
    # 骰子数量不能为负数；0 个骰子的总和就是 0。
    if num_dice < 0:
        raise ValueError("骰子数量不能为负数")

    total = 0
    rolled = 0
    while rolled < num_dice:
        total += random.randint(1, 6)
        rolled += 1
    return total


def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> Action:
    """3.根据双方生命值选择长门的行动。"""
    # 判断顺序很重要：长门快倒下时优先保命，其次才是收尾。
    if nagato_hp < NAGATO_LOW_HP_THRESHOLD:
        return "defend"
    elif nabiya_hp < NABIYA_SPECIAL_HP_THRESHOLD:
        return "special"
    else:
        return "attack"


def calculate_attack_damage(num_dice: int) -> int:
    """4.调用 roll_dice() 计算基础攻击伤害。"""
    return roll_dice(num_dice)


def calculate_defense_value(num_dice: int) -> int:
    """5.调用 roll_dice() 计算本回合的防御值。"""
    return roll_dice(num_dice)


def check_critical_hit(base_damage: int) -> bool:
    """6.判断基础伤害是否达到暴击阈值。"""
    return base_damage >= CRITICAL_HIT_THRESHOLD


def nabiya_ai_action(nabiya_hp: int) -> NabiyaAction:
    """7.根据娜比娅生命值选择她的行动。"""
    if nabiya_hp <= NABIYA_DEFEND_HP_THRESHOLD:
        return "defend"
    else:
        return "attack"


def calculate_final_damage(base_damage: int, defense_bonus: int) -> int:
    """8.用防御值抵消基础伤害，并返回不会小于零的最终伤害。"""
    # 伤害和防御值都不允许是负数。
    if base_damage < 0:
        raise ValueError("基础伤害不能为负数")
    if defense_bonus < 0:
        raise ValueError("防御值不能为负数")

    # 防御值最多把伤害抵消到 0，不能变成治疗。
    return max(0, base_damage - defense_bonus)


def apply_damage(current_hp: int, base_damage: int, defense_bonus: int = 0) -> int:
    """9.结算一次攻击并返回不会小于零的剩余生命值。"""
    final_damage = calculate_final_damage(base_damage, defense_bonus)

    remaining_hp = current_hp - final_damage
    return max(0, remaining_hp)


def is_battle_over(nagato_hp: int, nabiya_hp: int) -> bool:
    """10.判断是否至少有一名角色的生命值归零。"""
    return nagato_hp <= 0 or nabiya_hp <= 0


def get_battle_result(nagato_hp: int, nabiya_hp: int) -> BattleResult:
    """11.根据双方剩余生命值返回胜者或平局。"""
    if nagato_hp > 0 and nabiya_hp <= 0:
        return "nagato"
    if nabiya_hp > 0 and nagato_hp <= 0:
        return "nabiya"
    return "draw"


def main_battle_loop(
    pause_seconds: float = 0.0,
    max_turns: int = MAX_BATTLE_TURNS,
) -> BattleResult:
    """运行完整战斗，并返回 nagato、nabiya 或 draw。"""
    # 参数不合法时直接报错，避免出现死循环或负数的等待时间。
    if pause_seconds < 0:
        raise ValueError("pause_seconds 不能为负数")
    if max_turns <= 0:
        raise ValueError("max_turns 必须大于 0")

    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP

    # 防御值只有在被攻击时才会用到，抵消一次攻击后清零。
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0

    turn = 1

    print("=== 娜比娅偷吃事件：长门 vs 娜比娅 ===")
    print("长门：「汝，竟敢在余之港区放肆！」")

    while nagato_hp > 0 and nabiya_hp > 0 and turn <= max_turns:
        print(f"\n--- 第 {turn} 回合 ---")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        # ---------------- 长门的回合 ----------------
        action = choose_nagato_action(nagato_hp, nabiya_hp)

        if action == "attack":
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)

            # BIG SEVEN：基础伤害达到阈值就翻倍，再交给防御值抵消。
            if check_critical_hit(base_damage):
                base_damage *= 2
                print(f"长门触发「BIG SEVEN」，基础伤害翻倍为 {base_damage} 点！")
            else:
                print(f"长门发动炮击，基础伤害 {base_damage} 点。")

            old_hp = nabiya_hp
            nabiya_hp = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
            print(f"娜比娅受到 {old_hp - nabiya_hp} 点伤害。")

            # 攻击结束后，娜比娅的防御值已经被消耗掉。
            nabiya_defense_bonus = 0

        elif action == "defend":
            nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
            print(f"长门展现神子威仪，获得 {nagato_defense_bonus} 点防御值。")

        else:  # action == "special"
            if random.random() < SPECIAL_ATTACK_SUCCESS_RATE:
                old_hp = nabiya_hp
                nabiya_hp = apply_damage(
                    nabiya_hp, SPECIAL_ATTACK_DAMAGE, nabiya_defense_bonus
                )
                print(f"「四万神的守护」成功！娜比娅受到 {old_hp - nabiya_hp} 点伤害。")
            else:
                print("长门：「唔…失手了…」")
            nabiya_defense_bonus = 0

        # 长门行动后，如果娜比娅已经倒下，就不再进入娜比娅的回合。
        if is_battle_over(nagato_hp, nabiya_hp):
            break

        # ---------------- 娜比娅的回合 ----------------
        enemy_action = nabiya_ai_action(nabiya_hp)

        if enemy_action == "attack":
            base_damage = calculate_attack_damage(NABIYA_ATTACK_DICE)
            print(f"娜比娅发起反击，基础伤害 {base_damage} 点。")

            old_hp = nagato_hp
            nagato_hp = apply_damage(nagato_hp, base_damage, nagato_defense_bonus)
            print(f"长门受到 {old_hp - nagato_hp} 点伤害。")

        else:  # enemy_action == "defend"
            nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)
            print(f"娜比娅察觉危险，利用地形躲闪，获得 {nabiya_defense_bonus} 点防御值。")

        # 娜比娅行动结束后，长门这一回合的防御值已经失效。
        nagato_defense_bonus = 0

        turn += 1

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    # ---------------- 战斗结束 ----------------
    print("\n=== 战斗结束 ===")
    display_status("长门", nagato_hp, NAGATO_MAX_HP)
    display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

    result = get_battle_result(nagato_hp, nabiya_hp)

    if result == "nagato":
        print("长门获胜，娜比娅被赶跑了！")
    elif result == "nabiya":
        print("娜比娅获胜，长门没能拦住偷吃的塞壬……")
    else:
        print("战斗以平局收场。")

    return result
