"""娜比娅偷吃事件：回合制战斗练习。"""

from __future__ import annotations

from typing import Literal
import random, time

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
    if max_hp > 0:
        print(f"[{character_name}]HP: {current_hp} / {max_hp}")
    else:
        raise ValueError("最大生命值需大于0")


def roll_dice(num_dice: int) -> int:
    dianshu = 0
    if num_dice == 2 or num_dice == 3:
        i = 0
        while i < num_dice:
            dianshu += random.randint(1, 6)
            i += 1
        return dianshu
    else:
        raise ValueError("骰子数量只能为2或3")


def choose_nagato_action(nagato_hp: int, nabiya_hp: int) -> Action:
    if nagato_hp < NAGATO_LOW_HP_THRESHOLD:
        return "defend"
    elif (
        nagato_hp >= NAGATO_LOW_HP_THRESHOLD and nabiya_hp < NABIYA_SPECIAL_HP_THRESHOLD
    ):
        return "special"
    else:
        return "attack"


def calculate_attack_damage(num_dice: int) -> int:
    return roll_dice(num_dice)


def calculate_defense_value(num_dice: int) -> int:
    return roll_dice(num_dice)


def check_critical_hit(base_damage: int) -> bool:
    return base_damage >= CRITICAL_HIT_THRESHOLD


def nabiya_ai_action(nabiya_hp: int) -> NabiyaAction:
    if nabiya_hp <= NABIYA_DEFEND_HP_THRESHOLD:
        return "defend"
    else:
        return "attack"


def calculate_final_damage(base_damage: int, defense_bonus: int) -> int:
    if base_damage < 0:
        raise ValueError("基础攻击不能为负数")
    if defense_bonus < 0:
        raise ValueError("防御值不能为负数")
    return max(0, base_damage - defense_bonus)


def apply_damage(current_hp: int, base_damage: int, defense_bonus: int = 0) -> int:
    fshanghai = calculate_final_damage(base_damage, defense_bonus)
    return max(0, current_hp - fshanghai)


def is_battle_over(nagato_hp: int, nabiya_hp: int) -> bool:
    return nagato_hp <= 0 or nabiya_hp <= 0


def get_battle_result(nagato_hp: int, nabiya_hp: int) -> BattleResult:
    if nagato_hp == 0 and nabiya_hp == 0:
        return "draw"
    elif nagato_hp >= 0 and nabiya_hp == 0:
        return "nagato"
    elif nagato_hp == 0 and nabiya_hp >= 0:
        return "nabiya"
    else:
        return "draw"


def main_battle_loop(
    pause_seconds: float = 0.0,
    max_turns: int = MAX_BATTLE_TURNS,
) -> BattleResult:
    if pause_seconds < 0:
        raise ValueError("暂停时间不能小于0")
    if max_turns <= 0:
        raise ValueError("最大回合数必须大于0")
    nagato_hp = NAGATO_MAX_HP
    nabiya_hp = NABIYA_MAX_HP
    nagato_defense_bonus = 0
    nabiya_defense_bonus = 0
    turn = 1
    while nagato_hp > 0 and nabiya_hp > 0 and turn <= max_turns:
        print(f"第 {turn} 回合")
        display_status("nagato", nagato_hp, NAGATO_MAX_HP)
        display_status("nabiya", nabiya_hp, NABIYA_MAX_HP)
        action = choose_nagato_action(nagato_hp, nabiya_hp)
        if action == "attack":
            base_damage = calculate_attack_damage(3)
            if check_critical_hit(base_damage):
                base_damage *= 2
            nabiya_hp = apply_damage(nabiya_hp, base_damage, nabiya_defense_bonus)
            print(
                f"长门造成了{calculate_final_damage(base_damage,nabiya_defense_bonus)}点伤害,娜比娅生命值还剩{nabiya_hp}"
            )
            nabiya_defense_bonus = 0
        if action == "defend":
            nagato_defense_bonus = calculate_defense_value(2)
            print(f"长门获得了{nagato_defense_bonus}点防御值")
        if action == "special":
            if random.random() <= SPECIAL_ATTACK_SUCCESS_RATE:
                nabiya_hp = apply_damage(
                    nabiya_hp, SPECIAL_ATTACK_DAMAGE, nabiya_defense_bonus
                )
                print(
                    f"娜比娅发动了特殊攻击,造成{calculate_final_damage(SPECIAL_ATTACK_DAMAGE,nabiya_defense_bonus)}点伤害"
                )
            else:
                print(f"娜比娅的特殊攻击没有奏效")
            nabiya_defense_bonus = 0
        if is_battle_over(nagato_hp, nabiya_hp):
            break
        enemy_action = nabiya_ai_action(nabiya_hp)
        if enemy_action == "attack":
            base_damage = calculate_attack_damage(3)
            nagato_hp = apply_damage(nagato_hp, base_damage, nagato_defense_bonus)
            print(
                f"娜比娅发动攻击，造成了{calculate_final_damage(base_damage,nagato_defense_bonus)}点伤害"
            )
            print(f"长门生命值还剩{nagato_hp}")
            nagato_defense_bonus = 0
        else:
            nabiya_defense_bonus = calculate_defense_value(2)
        turn += 1
        time.sleep(pause_seconds)
    result = get_battle_result(nagato_hp, nabiya_hp)
    if result == "nagato":
        print("长门赢了")
    elif result == "nabiya":
        print("娜比娅赢了")
    else:
        print("平局")
    return result
