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
    if max_hp <= 0:
        raise ValueError("最大生命值必须大于 0")
    shown_hp = max(0, min(current_hp, max_hp))
    print(f"[{character_name}]HP: {shown_hp} / {max_hp}")


def roll_dice(num_dice: int) -> int:
    """投掷指定数量的六面骰子并返回点数总和。"""
    if num_dice < 0:
        raise ValueError("骰子数量不能为负数")

    total = 0
    rolled = 0
    while rolled < num_dice:
        total += random.randint(1, 6)
        rolled += 1
    return total


def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> Action:
    """根据双方生命值选择长门的行动。"""
    if nagato_hp < NAGATO_LOW_HP_THRESHOLD:
        return "defend"
    if nabiya_hp < NABIYA_SPECIAL_HP_THRESHOLD:
        return "special"
    return "attack"


def calculate_attack_damage(num_dice: int) -> int:
    """调用 roll_dice() 计算基础攻击伤害。"""
    return roll_dice(num_dice)


def calculate_defense_value(num_dice: int) -> int:
    """调用 roll_dice() 计算本回合的防御值。"""
    return roll_dice(num_dice)


def check_critical_hit(base_damage: int) -> bool:
    """判断基础伤害是否达到暴击阈值。"""
    return base_damage >= CRITICAL_HIT_THRESHOLD


def nabiya_ai_action(nabiya_hp: int) -> NabiyaAction:
    """根据娜比娅生命值选择她的行动。"""
    if nabiya_hp <= NABIYA_DEFEND_HP_THRESHOLD:
        return "defend"
    return "attack"


def calculate_final_damage(base_damage: int, defense_bonus: int) -> int:
    """用防御值抵消基础伤害，并返回不会小于零的最终伤害。"""
    if base_damage < 0 or defense_bonus < 0:
        raise ValueError("伤害和防御值不能为负数")
    return max(0, base_damage - defense_bonus)


def apply_damage(current_hp: int, base_damage: int, defense_bonus: int = 0) -> int:
    """结算一次攻击并返回不会小于零的剩余生命值。"""
    final_damage = calculate_final_damage(base_damage, defense_bonus)
    return max(0, current_hp - final_damage)


def is_battle_over(nagato_hp: int, nabiya_hp: int) -> bool:
    """判断是否至少有一名角色的生命值归零。"""
    return nagato_hp <= 0 or nabiya_hp <= 0


def get_battle_result(nagato_hp: int, nabiya_hp: int) -> BattleResult:
    """根据双方剩余生命值返回胜者或平局。"""
    if nagato_hp > 0 and nabiya_hp <= 0:
        return "nagato"
    if nagato_hp <= 0 and nabiya_hp > 0:
        return "nabiya"
    return "draw"


def main_battle_loop(
    pause_seconds: float = 0.0,
    max_turns: int = MAX_BATTLE_TURNS,
) -> BattleResult:
    """运行完整战斗，并返回 nagato、nabiya 或 draw。"""
    if pause_seconds < 0:
        raise ValueError("暂停时间不能为负数")
    if max_turns <= 0:
        raise ValueError("最大回合数必须大于 0")

    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1

    while not is_battle_over(nagato_hp, nabiya_hp) and turn <= max_turns:
        print(f"\n===== 第 {turn} 回合 =====")
        display_status("长门", nagato_hp, NAGATO_MAX_HP)
        display_status("娜比娅", nabiya_hp, NABIYA_MAX_HP)

        print("\n>>> 长门的回合")
        action = choose_nagato_action(nagato_hp, nabiya_hp)
        if action == "attack":
            base_damage = calculate_attack_damage(NAGATO_ATTACK_DICE)
            if check_critical_hit(base_damage):
                base_damage *= 2
                print("BIG SEVEN！炮击触发暴击，伤害翻倍！")
            hp_after = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
            print(f"长门发动炮击，造成 {nabiya_hp - hp_after} 点伤害！")
            nabiya_hp = hp_after
            nabiya_defense_bonus = 0
        elif action == "defend":
            nagato_defense_bonus = calculate_defense_value(NAGATO_DEFEND_DICE)
            print(f"长门展开威仪，获得 {nagato_defense_bonus} 点防御值！")
        else:
            if random.random() < SPECIAL_ATTACK_SUCCESS_RATE:
                hp_after = apply_damage(
                    nabiya_hp,
                    SPECIAL_ATTACK_DAMAGE,
                    nabiya_defense_bonus,
                )
                print(
                    "「四万神的守护」发动成功，"
                    f"造成 {nabiya_hp - hp_after} 点伤害！"
                )
                nabiya_hp = hp_after
            else:
                print("「唔…失手了…」特殊攻击没有造成伤害。")
            nabiya_defense_bonus = 0

        if is_battle_over(nagato_hp, nabiya_hp):
            break

        print("\n>>> 娜比娅的回合")
        enemy_action = nabiya_ai_action(nabiya_hp)
        if enemy_action == "attack":
            base_damage = calculate_attack_damage(NABIYA_ATTACK_DICE)
            old_hp = nagato_hp
            nagato_hp = apply_damage(nagato_hp, base_damage, nagato_defense_bonus)
            print(f"娜比娅发动反击，造成 {old_hp - nagato_hp} 点伤害！")
        else:
            nabiya_defense_bonus = calculate_defense_value(NABIYA_DEFEND_DICE)
            print(f"娜比娅利用地形逃窜，获得 {nabiya_defense_bonus} 点防御值！")

        nagato_defense_bonus = 0
        turn += 1
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    result = get_battle_result(nagato_hp, nabiya_hp)
    if result == "nagato":
        print("\n娜比娅落荒而逃，长门守护了港区的军粮！")
    elif result == "nabiya":
        print("\n长门没能挡住娜比娅，军粮被偷吃了……")
    else:
        print("\n战斗未分出胜负，双方暂时停手。")
    return result
