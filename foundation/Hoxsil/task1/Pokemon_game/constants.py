# ========== 概率常量 单位：百分比(0‑100) ==========
PARALYSIS_TRIGGER_CHANCE:int = 15       # 麻痹被动触发概率 %
THUNDERBOLT_PARALYSIS_CHANCE:int = 10   # 十万伏特麻痹附加概率 %
EMBER_BURN_CHANCE:int = 10            # 火花烧伤概率 %
FLAME_CHARGE_BURN_CHANCE:int = 80       # 蓄能爆炎烧伤概率 %
SEED_BOMB_POISON_CHANCE:int = 15        # 种子炸弹中毒概率 %
WATER_PASSIVE_REDUCE_CHANCE:int = 50    # 水被动减伤触发概率 %
HYDRO_PUMP_DEBUFF_CHANCE:int = 20       # 水炮闪避降低概率 %
QUICK_ATTACK_DOUBLE_HIT_CHANCE:int = 10 # 电光一闪双重攻击概率 %

# ========== 系数常量 ==========
GRASS_PASSIVE_HEAL_RATIO:float = 0.1      # 草被动回复比例
POISON_DAMAGE_RATIO:float = 0.1           # 中毒扣血比例
SEED_SUCK_RATIO:float = 0.1               # 寄生种子吸取比例
BURN_FIX_DAMAGE:float = 10                # 烧伤固定伤害
WATER_PASSIVE_REDUCE_RATIO:float = 0.3    # 水被动减伤
SHIELD_REDUCE_RATIO:float = 0.7           # 护盾减伤
FLAME_CHARGE_TMP_DOD:int = 20             # 蓄能爆炎判定时，对手临时加的闪避
THUNDERBOLT_DAMAGE_RATIO:float = 1.4      # 十万伏特伤害系数
QUICK_ATTACK_DAMAGE_RATIO:float = 1.0     # 电光一闪伤害系数
SEED_BOMB_DAMAGE_RATIO:float = 1.0        # 种子炸弹伤害系数
AQUA_JET_DAMAGE_RATIO:float = 1.4         # 水枪伤害系数
EMBER_DAMAGE_RATIO:float = 1.0            # 火花伤害系数
FLAME_CHARGE_DAMAGE_RATIO:float = 3.0     # 蓄能爆炎伤害系数
HYPER_BEAM_DAMAGE_RATIO:float = 1.8       # 破坏光线伤害系数
HYDRO_PUMP_DAMAGE_RATIO:float = 1.4       # 水炮伤害系数

# ========== 属性克制字典 ==========
# 克制返回 2，被克制返回 0.5，无影响返回 1
attribute_restraint = {
    ("grass", "grass"): 1,
    ("grass", "fire"): 0.5,
    ("grass", "water"): 2,
    ("grass", "electric"): 0.5,

    ("fire", "grass"): 2,
    ("fire", "fire"): 1,
    ("fire", "water"): 0.5,
    ("fire", "electric"): 1,

    ("water", "grass"): 0.5,
    ("water", "fire"): 2,
    ("water", "water"): 1,
    ("water", "electric"): 0.5,

    ("electric", "grass"): 2,
    ("electric", "fire"): 1,
    ("electric", "water"): 2,
    ("electric", "electric"): 1,
}
