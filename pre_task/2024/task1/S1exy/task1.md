# 宝可梦回合制对战游戏


## tips:

此作品基于西二ai task1 任务考核文档进行创作，没有使用参考文档（写完了才发现还有参考文档可以参考）并调整了部分的游戏功能和不合理的数值（没错说的就是你，妙蛙种子 ） 总体来说 可能 是依托屎山 不过能跑？！

玩家选择3名宝可梦，电脑随机选择1名宝可梦进行对战（有一些小小的bug就让电脑和玩家不可以选同一个宝可梦），游戏开始时候从3个宝可梦中选择一个出战，每个宝可梦拥有以下特性：

1. HP：此宝可梦的血量，当宝可梦的剩余HP小于等于0时，该宝可梦会【昏厥】
2. 属性： 宝可梦的属性会影响属性和抗性，相同属性的宝可梦会有一个相同的属性被动，宝可梦的属性会影响造成和受到的伤害
3. 攻击力：决定其对其他宝可梦的伤害
4. 防御力：来自其他宝可梦的伤害需要减去防御力的数值
5. 闪避率：在战斗中成功躲闪开敌人攻击的概率

| 属性 | 克制 | 被克制 | 属性被动                             |
| ---- | ---- | ------ | ------------------------------------ |
| 草   | 水   | 火     | 每回合回复 10% 最大HP值              |
| 火   | 草   | 水     | 每次造成伤害，叠加10%攻击力，最多4层 |
| 水   | 火   | 电     | 受到伤害时，有50%的几率减免30%的伤害 |
| 电   | 水   | 草     | 当成功躲闪时，可以立即使用一次技     |

被克制的宝可梦受到来自克制的宝可梦的伤害翻倍，被克制的宝可梦对克制的宝可梦造成的伤害减半

1. 招式：即技能，用来攻击对手的宝可梦，或者给对手的宝可梦附加各种各样的"效果"

## 4个宝可梦：

### 1.皮卡丘（PikaChu)

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YzkyZDg1NzMwZGRiOTBkYjE1MzhiNDhiYzMwODA3YmFfczIzYlpiT1h2ZlN6VHBEaERueDV4TDhqU1Azb2NrcFhfVG9rZW46SXQ1V2JVcmoyb2o4OXV4NzdzemNVRTh2bjdnXzE3Mjk1MDk4Mjk6MTcyOTUxMzQyOV9WNA)

**HP**: 80 **攻击力**: 35 **防御力**: 5 **属性**: 电 **躲闪率**: 30%

**十万伏特 (Thunderbolt)：**对敌人造成 1.4 倍攻击力的电属性伤害，并有 10% 概率使敌人麻痹

**电光一闪 (Quick Attack)：**对敌人造成 1.0 倍攻击力的快速攻击（快速攻击有几率触发第二次攻击），10% 概率触发第二次

### 2.**妙蛙种子 (Bulbasaur)**

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWQyMTY5ODk5YzMwN2U3YjRiNjJkZTg4ZWE5ZTRkMDNfRFg4YXRNS3R5R1Z2OUt2QzJHTk1NaDVuUDVnWlFYak1fVG9rZW46TnVGYmJ3QWxhb2tQRFN4Z3BrcmN4U3J1bjhlXzE3Mjk1MDk4Mjk6MTcyOTUxMzQyOV9WNA)

**HP**: 100 ***80***   **攻击力**: 35   **防御力**: 10   **属性**: 草   **躲闪率**: 10%

**种子炸弹 (Seed Bomb)：**妙蛙种子发射一颗种子，爆炸后对敌方造成草属性伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失 10% 生命值

**寄生种子 (Parasitic Seeds)：**妙蛙种子向对手播种，每回合吸取对手 10% ***5%*** 的最大生命值并恢复自己, 效果持续3回合

**！！！**削弱紧急削弱！！！

### 3.**杰尼龟（Squirtle）**

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=OTFjNzdiNDlhYjQ1OTZiN2Q5YTUzM2E3NGU5OGNlOGNfN1Z0VGkxMWk0d05EZ0ZFZ3diWkdESEFPU3JBMkJFWGtfVG9rZW46WGZ6WmJYZFRBb3dWM0J4RDQ4a2NtN2JGbkhjXzE3Mjk1MDk4Mjk6MTcyOTUxMzQyOV9WNA)

**HP**: 80 **攻击力**: 25 **防御力**: 20 **属性**: 水 **躲闪率**: 20%

**水枪 (Aqua Jet)：**杰尼龟喷射出一股强力的水流，对敌方造成 140% 水属性伤害

**护盾 (Shield)：**杰尼龟使用水流形成保护盾，减少下一回合受到的伤害50%  ***给自己增加10%的伤害抗性（超级耐打王）***

### 4.**小火龙（Charmander）**

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YzAxZmU2NjBiNzIxOTEyOTIyNGJjZmEzZmFiYjhmMjRfRnF4ejh0OVpDVXFrV0s2VnU1QjlidXd1YkZlZ1ZNN0pfVG9rZW46QjRubmJZYmthb0M3WXp4bXVpV2M1SzJZbmFjXzE3Mjk1MDk4Mjk6MTcyOTUxMzQyOV9WNA)

**HP**: 80 **攻击力**: 35 **防御力**: 15 **属性**: 火 **躲闪率**: 10%

**火花 (Ember)：**小火龙发射出一团小火焰，对敌人造成 100% 火属性伤害，并有10%的几率使目标陷入“烧伤”状态（每回合受到10额外伤害， 持续2回合）

**蓄能爆炎 (Flame Charge )：**小火龙召唤出强大的火焰，对敌人造成 300%  ***200%***  （太过逆天了，给对面秒了都必须削弱 ！）  火属性伤害，并有80%的几率使敌人陷入“烧伤”状态，这个技能需要1个回合的蓄力，并且在面对此技能时对方的闪避率增加 20%
