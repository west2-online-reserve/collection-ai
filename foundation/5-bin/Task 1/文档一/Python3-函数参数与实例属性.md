# Python3 函数参数、单独的 `*` 与实例属性

> 内容来源：学习宝可梦构造方法时的两个问题：“`*` 是什么意思？”以及“能否去掉 `*`，并且不初始化 `speed`？”
> 语义核对：[Python 官方文档：仅限关键字参数](https://docs.python.org/zh-cn/3.13/tutorial/controlflow.html#keyword-only-arguments)、[函数定义与默认值规则](https://docs.python.org/zh-cn/3.13/reference/compound_stmts.html#function-definitions)、[实例对象](https://docs.python.org/zh-cn/3.13/tutorial/classes.html#instance-objects)。
> 整理日期：2026-10-20。

## 作用

区分三个容易混在一起的问题：参数怎样传入、调用时能否省略，以及传入的值是否保存到对象中。函数和 `__init__()` 方法都遵循参数传递规则。

## 问题一：参数列表中单独的 `*` 是什么意思

提问中的签名片段如下。它需要放在类内部，且 `Skill` 须已定义或导入，不是独立脚本：

```python
def __init__(
    self,
    hp: int,
    attack: int,
    defense: int,
    dodge_chance: int = 0,
    charging_skill: Skill | None = None,
    extra_action_pending: bool = False,
    *,
    speed: int = 50,
) -> None:
    ...
```

单独的 `*` 是分界标记：**它后面的参数只能通过参数名传入**，称为“仅限关键字参数”。这里 `speed` 必须写成 `speed=80`，不能靠第几个位置来传。

`*` 本身不接收值，也不会改变后面参数的默认值。这个签名中，`hp` 等位于 `*` 前的参数既可以按位置传入，也可以按名称传入。

### 示例 1：必须按名称传速度，但可以省略

下面使用简化的独立类，只演示参数规则，不导入项目中的宝可梦类：

```python
class Fighter:
    def __init__(
        self,
        hp: int,
        attack: int,
        defense: int,
        dodge_chance: int = 0,
        *,
        speed: int = 50,
    ) -> None:
        self.dodge_chance = dodge_chance
        self.speed = speed

a = Fighter(100, 35, 10, speed=80)
b = Fighter(100, 35, 10)
c = Fighter(100, 35, 10, 80)

print(a.dodge_chance, a.speed)
print(b.dodge_chance, b.speed)
print(c.dodge_chance, c.speed)

try:
    Fighter(100, 35, 10, 0, 80)
except TypeError:
    print("速度不能通过位置传入")
```

输出：

```text
0 80
0 50
80 50
速度不能通过位置传入
```

第四个位置对应 `dodge_chance`，因此 `Fighter(100, 35, 10, 80)` 把闪避设为 80，速度仍用默认值 50。多塞一个位置参数给速度则会引发 `TypeError`。

## 问题二：能否去掉 `*`，并且“不初始化 speed”

可以去掉 `*`；但“不初始化”可能指**去掉参数默认值**，也可能指**不保存实例属性**，两者需要分别判断。

### 情况一：只去掉 `*`，保留 `speed=50`

此时 `speed` 也允许按位置传入，仍然可以写 `speed=80`，不传则使用默认值。位置传参时必须按完整参数顺序填写；在原签名中，速度排在闪避、蓄力技能和追加标记之后。

下面是缩短参数列表后的独立示例。

### 示例 2：去掉 `*` 后允许两种传法

```python
def stats(dodge_chance: int = 0, speed: int = 50) -> tuple[int, int]:
    return dodge_chance, speed

print(stats())
print(stats(0, 80))
print(stats(speed=80))
```

输出：

```text
(0, 50)
(0, 80)
(0, 80)
```

### 情况二：去掉 `*`，同时去掉 `speed` 的默认值

不能在原位置直接写成 `speed: int`：在 `*` 之前的参数区域，一旦出现有默认值的参数，后续普通参数也必须有默认值。原签名中的 `dodge_chance` 等已经有默认值。

这个错误在**定义函数时**发生，不需要等到调用。

### 示例 3：验证默认值顺序错误

```python
source = "def stats(dodge_chance: int = 0, speed: int):\n    pass\n"

try:
    compile(source, "<参数顺序示例>", "exec")
except SyntaxError:
    print("SyntaxError：没有默认值的普通参数排在了默认参数后面")
```

输出：

```text
SyntaxError：没有默认值的普通参数排在了默认参数后面
```

可以保留 `*`，让速度成为必须按名称提供的参数；也可以去掉 `*`，把无默认值的速度参数移到所有默认参数之前。两种方式会改变调用约定，需要同步核对调用处。

### 示例 4：让速度成为必填参数的两种方式

```python
def named_speed(dodge_chance: int = 0, *, speed: int) -> int:
    return speed

def first_speed(speed: int, dodge_chance: int = 0) -> int:
    return speed

print(named_speed(speed=80))
print(first_speed(80))

try:
    named_speed()
except TypeError:
    print("缺少必填的关键字参数 speed")
```

输出：

```text
80
80
缺少必填的关键字参数 speed
```

**仅限关键字不等于可选。** `speed: int = 50` 可以省略；`speed: int` 没有默认值，调用时必须提供。

### 情况三：不写 `self.speed = speed`

语法上可以。参数 `speed` 是方法内的局部名称，传入它不会自动为实例创建同名属性。要让对象之后保存并读取这个值，通常需要在初始化时赋值给 `self.speed`。

### 示例 5：参数和实例属性不是一回事

```python
class NotSaved:
    def __init__(self, speed: int = 50) -> None:
        print("方法内的 speed：", speed)

class Saved:
    def __init__(self, speed: int = 50) -> None:
        self.speed = speed

a = NotSaved(speed=80)
print("a 是否有 speed 属性：", hasattr(a, "speed"))

try:
    print(a.speed)
except AttributeError:
    print("AttributeError：没有可读取的 speed 属性")

b = Saved(speed=80)
print("保存后的 b.speed：", b.speed)
```

输出：

```text
方法内的 speed： 80
a 是否有 speed 属性： False
AttributeError：没有可读取的 speed 属性
保存后的 b.speed： 80
```

本例没有类属性、继承或其他赋值路径，所以读取 `a.speed` 会报错。实际项目中也可能从父类或类属性读到值，但这不等于传入的参数已经保存到了实例上。类型注解同样不会自动保存或赋值。

## 三种写法的区别

| 写法 | 负责什么 | 能否自动创建实例属性 |
| --- | --- | --- |
| `*, speed: int = 50` | 限制传入方式，并提供省略时的默认值 | 不能 |
| `*, speed: int` | 要求调用者按名称提供速度 | 不能 |
| 方法体内的 `self.speed = speed` | 把方法收到的值保存到当前实例 | 对本例的普通类可以 |

参数名和属性名也不必相同。例如 `self.speed = initial_speed` 保存的是参数 `initial_speed` 的值，后续通过实例的 `speed` 属性读取。

## 注意事项

- 参数表中单独的 `*` 是分界符，`*args` 则用于接收额外位置参数；两种形式都使后续具名参数只能通过关键字传入。
- 不要把“没有默认值”理解成“没有参数值”：无默认值的参数由调用者提供；有默认值的参数在省略时才使用默认值。
- 不要把“没有写 `self.speed`”理解成 Python 自动代为保存；需要明确对象之后从哪里读取速度。
- 本文只记录语法与参数设计讨论，不表示已经给游戏加入速度规则，也不修改项目构造方法。

## 示例验证与关联笔记

本文五个标有“输出”的独立示例已使用 `python3 -B` 逐段执行，实际输出与文中一致。提问中的原始签名是说明片段，不属于这五个独立示例。

- [类型提示与空值检查](<Python3-类型提示与空值检查.md>)
- [宝可梦对战阶段学习笔记](<Python3-宝可梦对战阶段学习笔记.md>)
- [property 与计算属性](<Python3-property与计算属性.md>)：解释 `display_name` 如何动态读取，以及只读属性与保存实例数据的区别。
