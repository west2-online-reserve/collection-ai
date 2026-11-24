# Python3 `dataclass` 数据类与冻结实例

> 内容来源：学习者对 `@dataclass(frozen=True)` 与 `SkillAction` 的提问。
> 语义核对：[Python 官方文档：dataclass](https://docs.python.org/zh-cn/3.13/library/dataclasses.html#dataclasses.dataclass)、[冻结的实例](https://docs.python.org/zh-cn/3.13/library/dataclasses.html#frozen-instances)、[FrozenInstanceError](https://docs.python.org/zh-cn/3.13/library/dataclasses.html#dataclasses.FrozenInstanceError)。
> 整理日期：2026-10-20。

## 作用

`dataclass` 适合定义以保存数据为主要用途的类。它根据字段声明生成初始化等方法，减少手动接收参数、保存属性的重复代码。

`frozen=True` 限制实例创建后的普通字段赋值和删除，适合表达一条创建后不应随意改写的动作记录。

## 提问中的代码

下面是讨论片段；使用前需要导入 `dataclass`，并让 `pokemon` 和 `Skill` 在注解所需的环境中可用，不是独立运行示例：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SkillAction:
    actor: pokemon.Pokemon
    skill: Skill
    is_player: bool
```

可以把它读成：“记录一只宝可梦准备使用的技能，以及它属于玩家还是电脑。”

| 字段 | 类型 | 保存的内容 |
| --- | --- | --- |
| `actor` | `pokemon.Pokemon` | 行动执行者的实例引用，不是受击目标 |
| `skill` | `Skill` | 本次选中的技能实例引用 |
| `is_player` | `bool` | 通常约定 `True` 表示玩家，`False` 表示电脑；含义由业务代码约定 |

字段没有默认值，所以创建时三个值都必须提供。该定义没有设置仅限关键字参数，既允许按字段顺序传入，也允许写出字段名；明确写出名称通常更好读。

## `@dataclass` 自动完成了什么

`@dataclass(...)` 是装饰器写法，它会处理下面定义的类。与普通类中单纯写类型注解不同，数据类会根据字段生成初始化逻辑，将传入值保存到实例属性。

默认生成的重要方法包括：

| 方法 | 用途 |
| --- | --- |
| `__init__()` | 按字段顺序接收参数，初始化实例字段 |
| `__repr__()` | 生成包含类名、字段名和值的调试表示 |
| `__eq__()` | 对同类型数据类实例逐个比较字段 |

这里比较的是字段各自的相等结果，不保证按宝可梦的名字或全部战斗数据比较；具体对象的比较行为取决于其类型。

### 示例 1：自动初始化、显示和比较

以下为独立示例，用字符串表示角色和技能名称，只演示数据类语法；项目中的原始定义保存的是对象引用。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SkillAction:
    actor: str
    skill: str
    is_player: bool

action = SkillAction(actor="妙蛙种子", skill="种子炸弹", is_player=True)
same = SkillAction("妙蛙种子", "种子炸弹", True)

print(action.actor)
print(action)
print(action == same)

try:
    SkillAction(actor="妙蛙种子", skill="种子炸弹")
except TypeError:
    print("缺少 is_player")
```

输出：

```text
妙蛙种子
SkillAction(actor='妙蛙种子', skill='种子炸弹', is_player=True)
True
缺少 is_player
```

## `frozen=True` 限制了什么

创建实例时可以正常初始化字段；创建后直接给字段重新赋值或删除字段，会引发 `FrozenInstanceError`。这不是说冻结实例不能初始化，而是生成的初始化逻辑有专门的赋值方式。

### 示例 2：创建后不能直接改写字段

```python
from dataclasses import dataclass, FrozenInstanceError

@dataclass(frozen=True)
class SkillAction:
    actor: str
    skill: str
    is_player: bool

action = SkillAction("妙蛙种子", "种子炸弹", True)

try:
    action.is_player = False
except FrozenInstanceError:
    print("不能重新赋值 is_player")

try:
    del action.actor
except FrozenInstanceError:
    print("不能删除 actor")

print(action.actor, action.is_player)
```

输出：

```text
不能重新赋值 is_player
不能删除 actor
妙蛙种子 True
```

同理，原始定义中的 `action.actor = 另一只宝可梦` 和 `action.skill = 另一个技能` 都是改写动作字段，会被阻止。需要表达另一条动作时，可以创建新的 `SkillAction`。

## 冻结记录不等于冻结引用的对象

原始定义中的 `actor` 和 `skill` 保存的是对象引用，创建记录不会自动复制对象。不能更换 `action.actor` 指向的对象，但该对象本身仍可以变化。

### 示例 3：对象仍会扣血，记录不是历史快照

这个独立示例使用简化的角色与技能类，不导入或修改游戏源码。

```python
from dataclasses import dataclass

class Actor:
    def __init__(self, hp: int) -> None:
        self.hp = hp

    def lose_hp(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)

class Skill:
    def __init__(self, name: str) -> None:
        self.name = name

@dataclass(frozen=True)
class SkillAction:
    actor: Actor
    skill: Skill
    is_player: bool

actor = Actor(hp=100)
skill = Skill("利爪")
action = SkillAction(actor=actor, skill=skill, is_player=True)

print(action.actor is actor)
print(actor.hp)  # 创建记录不会自动执行技能
actor.lose_hp(10)
print(action.actor.hp)
action.actor.lose_hp(5)
print(actor.hp)
skill.name = "演示改名"
print(action.skill.name)
```

输出：

```text
True
100
90
85
演示改名
```

`action.actor` 与 `actor` 指向同一个对象，所以无论从哪个引用扣血，另一处读取都会看到新血量。`skill` 对象的属性变化也会反映到 `action.skill` 上。

因此，这条记录不是完整的历史战况快照。需要保存行动发生时的血量时，应另行记录当时的数值，不能只保存宝可梦引用。

## 类型注解与业务校验仍需区分

字段注解描述期望的类型，供阅读和静态检查使用；普通数据类不会自动按这些注解检查传入值的类型，也不会自动校验游戏规则。

### 示例 4：数据类不会自动检查布尔类型

下面故意传入错误类型，用于演示运行时行为，不是推荐用法；类型检查器可能会报告这处调用。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Marker:
    is_player: bool

marker = Marker(is_player="玩家")
print(type(marker.is_player).__name__)
```

输出：

```text
str
```

原始 `SkillAction` 也不会自动保证技能属于执行者、执行者仍存活，或者阵营标记与执行者一致。这些应由游戏的决策与执行流程判断。

## 动作记录与执行流程的关系

创建 `SkillAction(...)` 只是保存选择，本身不攻击。后续执行流程才会读取 `action.actor` 和 `action.skill`，确定对手并调用技能。

这三个字段没有记录目标、伤害或最终结果；仅凭这段类定义，不能推断它已经实现伤害结算、行动排序或重放。本文解释提问中的数据结构，不声明项目已接入这些功能。

## 注意事项

- `frozen=True` 限制的是这条记录的普通属性赋值，不是对它引用的全部对象递归冻结。
- `dataclass` 是 Python 标准库功能，使用前要导入；三个字段的名称不是 Python 保留字。
- 具体子类实例可以作为父类类型字段的值，例如把妙蛙种子实例传给 `pokemon.Pokemon` 类型的 `actor`。
- 用对象引用记录“谁行动”，与另存数值记录“当时是什么状态”，用途不同。

## 示例验证与关联笔记

本文四个标有“输出”的独立示例已使用 `python3 -B` 逐段执行，实际输出与文中一致；提问中的原始项目片段不属于这四个独立示例。这里只保存笔记，没有修改业务代码、测试或项目配置。

- [函数参数、单独的 `*` 与实例属性](<Python3-函数参数与实例属性.md>)
- [类型提示与空值检查](<Python3-类型提示与空值检查.md>)
- [赋值与浅深复制](<Python3-赋值与浅深复制.md>)
