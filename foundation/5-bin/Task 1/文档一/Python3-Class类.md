# Python3 Class（类）

> 内容来源：Task 0 商品类、Task 1 校园角色与宝可梦实例。
> 语义核对：[Python 官方文档：类对象与实例对象](https://docs.python.org/3.13/tutorial/classes.html#class-objects)、[类变量和实例变量](https://docs.python.org/3.13/tutorial/classes.html#class-and-instance-variables)。
> 整理日期：2026-10-21。

## 作用

类将一类对象的数据和操作组织在一起，实例则是具体对象。例如商品类描述商品需要名称、价格等信息，某一本书是其中的一个实例。

宝可梦的 HP、技能和状态属于具体成员；不能因为两个成员来自同一个类，就让它们共用同一份可变状态。

## 语法

以下为语法示意，不作为完整程序运行：

```python
class ClassName:
    def __init__(self, value):
        self.value = value

    def method(self, amount):
        ...

obj = ClassName(value)
obj.method(amount)
```

## 参数

- `ClassName`：类名，通常使用单词首字母大写的命名方式。
- `self`：实例方法接收当前实例的参数名；通过实例调用方法时，由 Python 自动传入。
- `__init__()` 的其他参数：用于初始化实例的数据。
- `self.value`：保存到实例上的属性；只有局部变量 `value` 不会自动成为实例属性。
- 方法的其他参数：本次操作所需的数据，例如扣血量或礼物名称。

## 返回值

对本篇普通类调用 `ClassName(...)` 会得到新实例；`__init__()` 负责初始化已经创建的实例，应返回 `None`。

普通方法的返回值取决于其中的 `return`，不写返回值时为 `None`。修改实例属性与返回结果是两件不同的事。

## 示例 1：创建商品实例并计算销售收入

```python
class Product:
    def __init__(self, name: str, price: int, total: int, left: int) -> None:
        self.name = name
        self.price = price
        self.total = total
        self.left = left

    def income(self) -> int:
        return (self.total - self.left) * self.price

book = Product("笔记本", 10, 100, 30)
print(book.name)
print(book.income())
```

输出：

```text
笔记本
700
```

构造时传入的值被保存到 `book` 的属性中。计算收入时，方法读取的是当前实例的数据，不需要调用方再次传入同一组信息。这里为整数单价的教学简化，不展开金额精度与库存校验。

## 示例 2：self 指向本次调用的对象

```python
class Character:
    def __init__(self, name: str, hp: int) -> None:
        self.name = name
        self.hp = hp

    def lose_hp(self, amount: int) -> None:
        self.hp = max(0, self.hp - max(0, amount))

player = Character("玩家", 80)
computer = Character("电脑", 80)
player.lose_hp(20)

print(player.name, player.hp)
print(computer.name, computer.hp)
```

输出：

```text
玩家 60
电脑 80
```

调用 `player.lose_hp(20)` 时，`self` 指向 `player`，所以电脑的 HP 不变。调用方式可理解为 `Character.lose_hp(player, 20)`，但不应紧接着再执行一次，否则会重复扣血。

项目中技能方法的 `self` 是技能对象，受伤方法的 `self` 是受击宝可梦，需根据方法属于哪个类来判断。

## 示例 3：每个实例建立独立的技能列表

```python
class Player:
    role = "训练家"

    def __init__(self, name: str) -> None:
        self.name = name
        self.skills: list[str] = []

first = Player("甲")
second = Player("乙")
first.skills.append("火花")

print(first.role, second.role)
print(first.skills)
print(second.skills)
print(first.skills is second.skills)
```

输出：

```text
训练家 训练家
['火花']
[]
False
```

`role` 在类上定义，用来提供共同的身份信息；`skills` 在每次初始化时新建，因此两个实例的列表独立。如果把 `skills = []` 直接放到类上且不在实例中重新建立，就可能让多个对象共用这个列表。

## 注意事项

- `self` 是约定名称，不是关键字；按约定使用它更容易阅读。
- `obj.method` 取得绑定方法，`obj.method()` 才执行方法，缺少括号可能导致逻辑没有发生。
- 不要为实例独有的可变数据使用共享的类列表或可变默认参数。
- `__init__()` 不是用来返回新实例的方法，返回非 `None` 的值会报错。
- 参数与属性的详细区别见 [函数参数与实例属性](<Python3-函数参数与实例属性.md>)；职责设计见 [OOP 面向对象思想](<Python3-OOP面向对象思想.md>)。
