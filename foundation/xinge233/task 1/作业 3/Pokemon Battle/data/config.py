# 字符画
# -*- coding: utf-8 -*-
r"""
Pokemon_Battle 字符画（空心描边风格）
- 尺寸：6 行 x 119 列
- 笔画：只用 / \ _ | 描边，字心一律留空，不做任何填充
- 结构：每一行都是列表 Pokemon_Battle 的一个元素，for 循环逐行打印即可还原
- 提示：请用等宽字体（Consolas / 黑体等）查看，比例字体会导致错位
"""

Pokemon_Battle = [
    r" ____              _                                                       ____                              _",
    r"|  _ \            | |                                                     | __ )              _       _     | |",
    r"| |_) |    ___    | | _      ___    _ __ ___      ___     _ __            |  _ \     __ _    | |_    | |_   | |    ___",
    r"|  __/    / _ \   | |/ /    / _ \  | '_ ` _ \    / _ \   | '_ \           | |_) |   / _` |   | __|   | __|  | |   / _ "
    + "\\",
    r"| |      | (_) |  | |\ \   |  __/  | | | | | |  | (_) |  | | | |          |  __/   | (_| |   | |_    | |_   | |  |  __/",
    r"|_|       \___/   |_| \_\   \___|  |_| |_| |_|   \___/   |_| |_|  ______  |____/    \__,_|    \__|    \__|  |_|   \___|",
]
# -*- coding: utf-8 -*-
r"""
Win 字符画（空心描边风格）
- 尺寸：5 行 x 29 列
- 笔画：只用 / \ _ | 描边，字心一律留空，不做任何填充
- 结构：每一行都是列表 Win 的一个元素，for 循环逐行打印即可还原
- 提示：请用等宽字体（Consolas / 黑体等）查看，比例字体会导致错位
"""
Win = [
    r" __          __   _",
    r" \ \        / /  (_)   _ __",
    r"  \ \  /\  / /   | |  | '_ " + "\\",
    r"   \ \/  \/ /    | |  | | | |",
    r"    \_/\_/       |_|  |_| |_|",
]
# 错误输入提示
wrong_input = "请正确输入选项"
error_text = "在{location}出现问题"
# 阵营
player_side = "我方"
enemy_side = "敌方"
player_prefix = f"[{player_side}]"
enemy_prefix = f"[{enemy_side}]"
enermy_team_situation_text = f"[敌方]队伍的宝可梦为:"
# 通用 game.py

# reset

# confirm

confirm_text = "你确定要{act}吗"

# sure

sure_text = ["1.确认", "2.取消"]
sure_input = "请用1/2回答:"

# save
save_success = "文件已成功保存在{path}"
save_error = "保存出错，你的程序可能未保存"
# Game

# pick

num_role_relationship = {
    "1": "Pikachu",
    "2": "Bulbasaur",
    "3": "Squirtle",
    "4": "Charmander",
    "5": "MilkDragon",
}
organize_team_text = "请选择 3 个宝可梦用于组成你的队伍："
organize_input_text = "请用空格分割的1~5选择三个需要的角色:"
name_element_info = "{index}.{name} {element}属性"
element_passives_text = "{element}属性被动：{passive}"
element_passives = {
    "草": "每回合回复 10% 最大 HP",
    "火": "每次造成伤害后，攻击力叠加 10%，最多叠加 4 层",
    "水": "受到伤害时，有 50% 的概率减免 30% 的伤害",
    "电": "成功闪避攻击时，可以立即使用一次技能",
}
state_info = "最大生命值:{maxhp}，攻击力:{attack}，防御力:{denfense}，闪避率:{miss}%"
skill_info = "{name1}: {skill1}\n{name2}: {skill2}"
wrong_team_text = "我说选三个你耳朵聋吗？"
team_comfirm_text = "选择这三个角色"
team_roles = "请选择你要出战的宝可梦:\n{team_choices}"
fight_input_text = "请用1~{team_size}选择你要出战的宝可梦:"
pokemon_choose_conform_text = "你选择了{pokemon_name}，确定吗？"
enermy_team_text = f"{enemy_side}宝可梦为：{{enermy1}}、{{enermy2}}、{{enermy3}}"
# battle_ready
battle_ready_text = "遭遇了{enermy_pokemon}，准备战斗！"
# battle
battle_start_text = "------战斗开始------"
round_info = "第{round}回合"
act_tip_text = (
    "请选择你在第{round}回合的行动:\n1.普通攻击\n2.技能\n3.调查(不会消耗回合)\n4.投降"
)
battle_input_text = "请用1~4选择你要进行的行动:"
pokemon_situation_text = f"{player_side}宝可梦:{{pokemon_name}} {{hp}}/{{max_hp}} HP "
enermy_pokemon_situation_text = (
    f"{enemy_side}宝可梦:{{pokemon_name}} {{hp}}/{{max_hp}} HP "
)
winner_situation_text = "当前比分为{score1}:{socore2},当一方率先赢得两回合时游戏结束"
battle_end_text = "战斗结束，{pokemon}被击败了！"
# role.py

role_names = ["皮卡丘", "妙蛙种子", "杰尼龟", "小火龙", "奶龙"]
amount_up_text = "你的{name}提升了{amount}点"
amount_down_text = "你的{name}降低了{amount}点"
health_text = "生命值"
max_health_text = "最大生命值"
attack_text = "攻击力"
defense_text = "防御力"
miss_text = "闪避率"
water_passive_text = "水系{name}精灵触发被动，受到伤害降低30%"
electric_passive_text = "电系精灵闪避时可以额外使用一次技能"

# action.py
# display句式
round_count_text = "第{round}回合"
act_text = "{my_pokemon}对{enermy_pokemon}进行了{action_name}"
act_display_name = {"fight": "普通攻击", "skill": "技能", "insruction": "说明"}
miss_success_text = "{my_pokemon}闪避了{enermy_pokemon}的攻击"
counter_text = "效果拔群！"
be_counter_text = "好像没什么效果..."
electric_passive_success_text = "{my_pokemon}即将闪避{enermy_pokemon}的攻击\n电系{my_pokemon}闪避了{enermy_pokemon}的攻击，额外使用一次技能"
damage_text = "{my_pokemon}对{enermy_pokemon}造成了{damage}点伤害"
die_text = "{pokemon_name}被击败了！"
Pikachu_skill1_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 十万伏特"
Pikachu_skill2_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 电光一闪"
Pikachu_paralysis_text = "{enermy_pokemon}被麻痹了！"
Pikachu_skill2_success_text = "{my_pokemon}的电光一闪触发了第二次攻击！"
paralysis_text = "{pokemon}被麻痹了，无法动弹"
Bulbasaur_skill1_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 种子炸弹"
Bulbasaur_skill1_success_text = "{enermy_pokemon}被中毒了！"
poison_text = "{pokemon}因中毒而损失了10%的最大生命值"
poison_remain_round_text = "中毒状态还剩{round}回合"
Bulbasaur_skill2_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 寄生种子"
Bulbasaur_skill2_success_text = "{enermy_pokemon}被寄生了！"
Bulbasaur_skill2_fail_text = "寄生种子未命中{enermy_pokemon}！"
absorb_text = "{enermy_pokemon}吸收了{pokemon}最大生命的10%"
Charmander_skill1_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 火花"
Charmander_skill1_success_text = "{enermy_pokemon}陷入「烧伤」状态！"
burn_text = "{pokemon}因烧伤而损失了10点生命值"
burn_remain_round_text = "烧伤状态还剩{round}回合"
Charmander_skill2_text = "{pokemon}正在蓄力「蓄能爆炎」"
Flame_Charge_text = "{pokemon} 对 {enermy_pokemon} 使用了 蓄能爆炎"
Squirtle_skill1_text = "{my_pokemon} 对 {enermy_pokemon} 使用了 水枪"
Squirtle_skill2_text = "{pokemon} 使用了护盾，减少下一回合受到的伤害50%"
shield_text = "{pokemon}的护盾生效，减少了50%的伤害"
shield_disappear_text = "{pokemon}的护盾消失了"
MilkDragon_skill1_text = "{my_pokemon} 使用了 唐笑"
MilkDragon_skill1_confusion_text = "{my_pokemon}的唐笑使{enermy_pokemon}混乱了！"
MilkDragon_skill1_attack_down_text = (
    "{my_pokemon}的唐笑使{enermy_pokemon}攻击力下降了20%！持续 3 回合"
)
MilkDragon_skill1_defense_down_text = (
    "{my_pokemon}的唐笑使{enermy_pokemon}防御力下降了20%！持续 3 回合"
)
MilkDragon_skill1_stun_text = (
    "{my_pokemon}的唐笑使{enermy_pokemon}陷入了眩晕状态！持续 3 回合"
)
MilkDragon_skill1_nothing_text = "但是{enermy_pokemon}使用了轻松绷住"
attack_down_end_text = "{pokemon}的攻击力下降状态结束了"
defense_down_end_text = "{pokemon}的防御力下降状态结束了"
attack_up_end_text = "{pokemon}的攻击力提升状态结束了"
MilkDragon_skill2_text = "{my_pokemon} 向全世界宣告他才是奶龙!\n{my_pokemon}的攻击力提升了40%，且变成普通攻击附带电属性，持续 3 回合"
confusion_text="{pokemon}陷入了混乱状态，自己攻击了自己"
paralysis_success_text="{pokemon}陷入了麻痹状态，行动失败"
fire_passive_text="{pokemon}的火属性被动技能生效，增加了{attack}点攻击力"
grass_passive_text="{pokemon}的草属性被动技能生效，恢复了{heal}点生命值"
Flame_Charge_burn_text="{attacker}的蓄能爆炎使{defender}陷入了烧伤状态！"
Charging_text="{pokemon}正在蓄力「蓄能爆炎」"
stun_text="{pokemon}陷入了眩晕状态，无法行动"
# check
check_options = (
    f"1.查看{player_side}出战精灵情况\n"
    f"2.查看{enemy_side}精灵情报\n"
    "3.查看克制关系"
)
check_input_text = "请用1~3回答:"
check_pokemon_situation_text = f"{player_side}宝可梦:{{pokemon_name}} {{hp}}/{{max_hp}} HP 攻击力为{{attack}}，防御力为{{defense}}，闪避率为{{miss}}%"
check_enermy_pokemon_situation_text = f"{enemy_side}宝可梦:{{pokemon_name}} {{hp}}/{{max_hp}} HP 攻击力为{{attack}}，防御力为{{defense}}，闪避率为{{miss}}%"
# skill
skill_options_text = (
    "1.{skill1_name}:{skill1_description}\n2.{skill2_name}:{skill2_description}"
)
skill_options_input_text = "请用1~3回答:"
# surrender
victory_text = "恭喜你获得了胜利！"
surrender_text = "胜败乃兵家常事，大侠请重新来过"
end_options = "1.再打一回合！\n2.回到主菜单\n3.退出游戏"
end_input_text = "请用1~3回答:"
# 用于主菜单 start.py

# 用于在菜单选择时让用户输入对应数字
menu_choice_text = "请用1~5回答:"
# 用于退出时显示相应操作
quit_text = "退出游戏"

# 标题
title = "Pokemon Battle"

# 界面选项
main_menu_options = [
    "1.开始游戏",
    "2.退出游戏",
]


# 克制关系
effect_relationship = {
    "Grass": "Water",
    "Fire": "Grass",
    "Water": "Fire",
    "Electric": "Water",
}  # 键对值伤害翻倍
be_effect_relationship = {
    "Grass": "Fire",
    "Fire": "Water",
    "Water": "Electric",
    "Electric": "Grass",
}  # 键对值伤害减半

effect_relationship_text = (
    "克制关系图：\n"
    "[草] --克制--> [水] --克制--> [火] --克制--> [草]\n"
    "[电] --克制--> [水]"
)

pokemon_text_dict = {
    "Pikachu": {
        "id": "0",
        "max_hp": 80,
        "hp": 80,
        "attack": 35,
        "defense": 5,
        "type": "电",
        "miss": 30,
        "name": "皮卡丘",
        # "class":Pikachu(),
        "skill_1": {
            "name": "十万伏特",
            "description": "造成1.4倍攻击力的电属性伤害，10%概率使敌人麻痹3回合",
        },
        "skill_2": {
            "name": "电光一闪",
            "description": "造成0.9倍攻击力的伤害，10%概率触发第二次攻击",
        },
    },
    "Bulbasaur": {
        "id": "1",
        "max_hp": 100,
        "hp": 100,
        "attack": 35,
        "defense": 10,
        "type": "草",
        "miss": 10,
        "name": "妙蛙种子",
        # "class":Bulbasaur(),
        "skill_1": {
            "name": "种子炸弹",
            "description": "造成草属性伤害，15%概率使目标中毒，每回合损失10%生命值，持续3回合",
        },
        "skill_2": {
            "name": "寄生种子",
            "description": "50%的概率，每回合吸取对手10%的最大生命值并恢复自己，持续3回合",
        },
    },
    "Squirtle": {
        "id": "2",
        "max_hp": 80,
        "hp": 80,
        "attack": 25,
        "defense": 20,
        "type": "水",
        "miss": 20,
        "name": "杰尼龟",
        # "class":Squirtle(),
        "skill_1": {
            "name": "水枪",
            "description": "造成1.4倍攻击力的水属性伤害",
        },
        "skill_2": {
            "name": "护盾",
            "description": "减少下一回合受到的伤害50%",
        },
    },
    "Charmander": {
        "id": "3",
        "max_hp": 80,
        "hp": 80,
        "attack": 35,
        "defense": 15,
        "type": "火",
        "miss": 10,
        "name": "小火龙",
        # "class":Charmander(),
        "skill_1": {
            "name": "火花",
            "description": "造成1.0倍攻击力的火属性伤害，10%概率使目标烧伤",
        },
        "skill_2": {
            "name": "蓄能爆炎",
            "description": "需要蓄力1回合，造成3.0倍攻击力的火属性伤害，80%概率使目标烧伤",
        },
    },
    "MilkDragon": {
        "id": "4",
        "max_hp": 110,
        "hp": 110,
        "attack": 25,
        "defense": 20,
        "type": "电",
        "miss": 10,
        "name": "奶龙",
        # "class":MilkDragon(),
        "skill_1": {
            "name": "唐笑",
            "description": "有概率使敌人混乱、降低攻击、降低防御、眩晕",
        },
        "skill_2": {
            "name": "我才是奶龙！",
            "description": "攻击变为电属性且攻击力提升40%，持续3回合",
        },
    },
}  # ai整理
