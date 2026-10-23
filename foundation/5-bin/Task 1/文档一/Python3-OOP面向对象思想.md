# Python3 OOP（面向对象编程思想）

> 内容来源：校园角色、宝可梦角色/技能/效果的职责划分与阶段学习笔记。
> 语义核对：[Python 官方文档：继承](https://docs.python.org/3.13/tutorial/classes.html#inheritance)、[私有变量的约定](https://docs.python.org/3.13/tutorial/classes.html#private-variables)、[super()](https://docs.python.org/3.13/library/functions.html#super)。
> 整理日期：2026-10-21。

## 作用

面向对象编程把状态和操作状态的行为组织到对象中。重点不仅是写出 `class`，还包括确定谁保存数据、谁负责改变数据，以及对象如何协作。

宝可梦负责自己的 HP 边界，技能负责施放效果，游戏控制器负责回合安排，这样规则变化时更容易找到应该修改的位置。

## 语法

OOP 是组织代码的思想，没有一个单独的调用语法。以下只示意常见结构，不作为完整程序运行：

```python
class Base:
    def act(self):
        ...

class Child(Base):
    def act(self):
        ...

class Controller:
    def __init__(self, members):
        self.members = members
```

## 参数

这里说明设计中的构成要素，不是给 OOP 虚构一个统一参数列表：

| 要素 | 需要回答的问题 | 作业中的例子 |
| --- | --- | --- |
| 对象状态 | 数据由谁保存？ | 宝可梦保存 HP 和状态列表 |
| 方法参数 | 一次操作还需什么信息？ | 技能接收使用者和目标 |
| 继承关系 | 是否属于同一类对象？ | 妙蛙种子属于草属性宝可梦 |
| 组合关系 | 谁拥有谁？ | 宝可梦拥有技能对象 |
| 公共接口 | 调用方需要知道什么？ | 各技能都提供 `execute()` |

## 返回值

OOP 本身没有返回值。具体构造调用返回实例；查询方法可以返回计算结果；修改状态的方法可以返回 `None`。判断是否改变了状态，要看对象属性，不能只看方法返回值。

## 示例 1：封装生命值边界

```python
class Character:
    def __init__(self, hp: int) -> None:
        self.hp = max(0, hp)

    def lose_hp(self, amount: int) -> None:
        if amount <= 0 or self.hp == 0:
            return
        self.hp = max(0, self.hp - amount)

character = Character(80)
character.lose_hp(-10)
print(character.hp)
result = character.lose_hp(100)
print(character.hp)
print(result)
```

输出：

```text
80
0
None
```

扣血规则集中在一个入口：负数不变成治疗，超量伤害不使 HP 变成负数。实际宝可梦项目还维护 `alive`、防御、护盾等规则；本例只演示封装的作用。

封装不是仅给属性加下划线。Python 的常见做法是通过约定和方法接口维护规则，不能假定外部代码完全无法访问属性。

## 示例 2：继承公共数据，通过重写表现多态

```python
class Character:
    def __init__(self, name: str, max_hp: int) -> None:
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp

    def begin(self) -> None:
        pass

class GrassCharacter(Character):
    def __init__(self, name: str) -> None:
        super().__init__(name, 100)

    def begin(self) -> None:
        if self.hp > 0:
            self.hp = min(self.max_hp, self.hp + int(self.max_hp * 0.1))

grass = GrassCharacter("草属性成员")
grass.hp = 60
normal = Character("普通成员", 80)
for member in [grass, normal]:
    member.begin()
    print(member.name, member.hp)
```

输出：

```text
草属性成员 70
普通成员 80
```

`super().__init__()` 复用公共初始化。两个对象都能接收 `begin()` 调用，但草属性成员执行回血，普通成员使用基类的空行为，这就是本例的多态。

基类里的 `pass` 是允许无额外开始行为的接口设计，并非每处 `pass` 都代表未完成作业。示例把 HP 设为 `60` 只是准备受伤场景。

## 示例 3：组合技能对象，让双方通过接口协作

```python
class Skill:
    def __init__(self, name: str, damage: int) -> None:
        self.name = name
        self.damage = damage

    def execute(self, target: "Fighter") -> None:
        target.lose_hp(self.damage)

class Fighter:
    def __init__(self, name: str, hp: int, skills: list[Skill]) -> None:
        self.name = name
        self.hp = hp
        self.skills = skills.copy()

    def lose_hp(self, amount: int) -> None:
        self.hp = max(0, self.hp - max(0, amount))

    def use_skill(self, index: int, target: "Fighter") -> None:
        skill = self.skills[index]
        print(f"{self.name} 使用 {skill.name}")
        skill.execute(target)

player = Fighter("喵喵", 80, [Skill("利爪", 12)])
opponent = Fighter("对手", 80, [])
player.use_skill(0, opponent)
print(opponent.hp)
```

输出：

```text
喵喵 使用 利爪
68
```

角色“拥有”技能，角色不是技能的子类。技能通过目标的扣血方法改变状态，不需要直接维护目标的生命值边界。这里假定下标有效，伤害数值是教学简化，不包含项目中的防御、闪避与属性倍率。

本例复制了技能列表的外层，未复制其中的技能对象；不要把它理解为完整的队伍独立性方案。

## 注意事项

- 封装集中维护规则；继承复用共同接口；多态允许相同调用对应不同实现；组合让对象持有并使用其他对象。
- 不必为了复用几行代码建立很深的继承层次，先确认关系是否清楚。
- 游戏控制器安排回合，不应重复实现每种技能的具体效果；职责清晰比单纯减少行数更重要。
- Python 的多态不强制要求共同基类，提供兼容接口的对象也能协作。
- 方法名称相同不表示行为可以随意变化，仍要遵守调用方期待的输入、输出和状态规则。
- 项目细节见 [宝可梦阶段学习笔记](<Python3-宝可梦对战阶段学习笔记.md>)，语法基础见 [Class 类](<Python3-Class类.md>)。
