# 娜比娅偷吃事件

## 故事背景

长门发现塞壬人形娜比娅正在偷吃军粮。本作业需要补全战斗核心逻辑，帮助长门完成这场全自动的 1v1 回合制对决。

## 运行方法

在本目录执行：

```bash
python check_my_code.py
python start_the_battle.py
```

## 战斗规则

- 长门初始 HP 为 115，娜比娅初始 HP 为 105，HP 先归零者败。
- 长门普通攻击掷 3 个六面骰子；基础伤害达到 15 时触发「BIG SEVEN」，伤害翻倍。
- 长门防御掷 2 个骰子，所得防御值抵消娜比娅下一次攻击。
- 长门特殊攻击有 45% 概率造成固定 24 点伤害。
- 娜比娅普通攻击掷 3 个骰子，不会暴击；HP 小于等于 40 时选择防御并掷 2 个骰子。
- 长门 HP 小于 30 时优先防御；否则，娜比娅 HP 小于 20 时使用特殊攻击，其余情况普通攻击。
- 战斗最多进行 50 回合，达到上限且双方仍存活时为平局。

## 需要完成的 12 个函数

1. `display_status`：按 `[角色名]HP: 当前血量 / 最大血量` 输出状态。
2. `roll_dice`：用 `while` 循环投掷指定数量的六面骰子。
3. `choose_nagato_action`：根据双方 HP 选择长门行动。
4. `calculate_attack_damage`：调用 `roll_dice()` 计算基础伤害。
5. `calculate_defense_value`：调用 `roll_dice()` 计算防御值。
6. `check_critical_hit`：判断基础伤害是否达到 15。
7. `nabiya_ai_action`：根据娜比娅 HP 选择攻击或防御。
8. `calculate_final_damage`：扣除防御并拒绝负数参数。
9. `apply_damage`：结算伤害，保证 HP 不小于 0。
10. `is_battle_over`：判断任意一方是否倒下。
11. `get_battle_result`：返回 `nagato`、`nabiya` 或 `draw`。
12. `main_battle_loop`：运行完整回合循环并返回结果。
