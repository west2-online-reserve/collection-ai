# Pokemon Battle 宝可梦对战

## 项目介绍

这是一个使用 Python 编写的命令行回合制宝可梦对战游戏。玩家从 5 只宝可梦中选择 3 只不同的宝可梦组成队伍，电脑随机选择 3 只，直到一方队伍全部昏厥。

最低运行版本为 **Python 3.10+**。

## 游戏玩法

1. 按提示输入 3 个不同编号，组成玩家队伍。
2. 电脑随机组成 3 只宝可梦的队伍。
3. 玩家选择第一只出战宝可梦，电脑随机选择首发。
4. 每回合输入技能编号进行行动；电脑自动选择技能。
5. 玩家宝可梦昏厥后，从存活队员中选择替补；电脑会自动换上存活队员。
6. 一方队伍全部昏厥时结束；若双方同时全部昏厥，则判为平局。

## 宝可梦

| 宝可梦 | 属性 | HP | Attack | Defense | Dodge | 技能 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PikaChu | Electric | 80 | 35 | 5 | 30% | Thunderbolt、Quick Attack |
| Bulbasaur | Grass | 100 | 35 | 10 | 10% | Seed Bomb、Parasitic Seeds |
| Squirtle | Water | 80 | 25 | 20 | 20% | Aqua Jet、Shield |
| Charmander | Fire | 80 | 35 | 15 | 10% | Ember、Flame Charge |
| Chikorita | Grass | 90 | 30 | 15 | 15% | Seed Bomb、Parasitic Seeds |

Chikorita 是本项目的自定义宝可梦，可以正常加入队伍、出战并使用草属性被动和技能。

## 主要战斗规则

- 属性克制：Grass 克 Water、Fire 克 Grass、Water 克 Fire、Electric 克 Water，伤害为 2 倍；反向攻击伤害为 0.5 倍。
- Grass：每个回合结束时，仍存活的当前出战宝可梦恢复最大 HP 的 10%。
- Fire：每次技能成功造成伤害后，攻击力增加基础攻击力的 10%，最多 4 层。
- Water：受到普通伤害时有 50% 概率减免 30% 伤害。
- Electric：成功闪避后立即随机使用一个技能反击；反击不能产生新的反击链。
- Thunderbolt：造成 `1.4 × attack` 的电属性伤害，并有 10% 概率使目标麻痹。
- Quick Attack：造成 `1.0 × attack` 的伤害，并有 10% 概率进行第二次攻击。
- Seed Bomb：造成草属性伤害，并有 15% 概率使目标中毒；中毒持续 3 回合，每回合造成目标最大 HP 的 10% 伤害。
- Parasitic Seeds：持续 3 回合，每回合吸取目标最大 HP 的 10%，并为施放者恢复实际吸取的生命值。Chikorita 的自定义版本每回合吸取 8 点。
- Aqua Jet：造成 `1.4 × attack` 的水属性伤害。
- Shield：使下一次受到的普通伤害减少 50%，生效后消失。
- Ember：造成 `1.0 × attack` 的火属性伤害，并有 10% 概率烧伤。
- Flame Charge：第一回合蓄力，下一回合造成 `3.0 × attack` 的火属性伤害；有 80% 概率烧伤，目标对该技能的闪避率额外增加 20%。
- 烧伤持续 2 回合，每回合造成 10 点固定伤害。固定状态伤害不受普通防御、护盾或 Water 被动影响。
- 麻痹会使宝可梦下一次行动无法进行，玩家和电脑遵循相同规则。
- HP 始终不会显示为负数；昏厥的宝可梦不能继续行动或结算状态。
- 同类状态再次添加时会刷新为新状态，不会无限叠加。

## 文件结构

```text
Pokemon_Battle/
├── main.py              # 程序入口
├── game.py              # 队伍选择、回合流程、换人和胜负判断
├── pokemon.py           # 宝可梦、属性被动和基础战斗行为
├── skills.py            # 技能实现
├── effects.py           # 烧伤、中毒、治疗和寄生种子效果
├── README.md
├── test_chikorita.py    # Chikorita 初始化与草属性被动测试
├── test_electric.py     # Electric 闪避反击测试
├── test_fire.py         # Fire 被动与蓄力技能测试
└── test_status.py       # 状态、换人、胜负与完整战斗测试
```

## 运行方式

在项目目录运行：

```bash
python main.py
```

## 测试方法

可以逐个运行简单断言测试：

```bash
python test_chikorita.py
python test_electric.py
python test_fire.py
python test_status.py
```

如果环境安装了 pytest，也可以运行：

```bash
python -m pytest -q
```
