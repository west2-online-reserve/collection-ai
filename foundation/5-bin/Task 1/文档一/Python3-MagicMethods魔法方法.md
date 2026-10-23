# Python3 Magic Methods（魔法方法）

> 内容来源：Task 0 `MyZoo` 练习，以及宝可梦和技能对象的 `__str__()`。
> 语义核对：[Python 官方文档：特殊方法](https://docs.python.org/3.13/reference/datamodel.html#special-method-names)、[NotImplemented](https://docs.python.org/3.13/library/constants.html#NotImplemented)。
> 整理日期：2026-10-21。

## 作用

魔法方法也称特殊方法，名称通常以双下划线开头和结尾。它们让自定义对象参与 Python 已有的操作，例如打印、求长度、比较相等。

调用方可以使用 `str(obj)`、`len(obj)`、`a == b`，由类定义这些操作的具体含义。

## 语法

以下为语法示意，不作为完整程序运行：

```python
class Example:
    def __init__(self, value):
        ...

    def __str__(self):
        ...

    def __len__(self):
        ...

    def __eq__(self, other):
        ...
```

## 参数

- `self`：当前参与操作的实例。
- `other`：`__eq__()` 中拿来与当前实例比较的另一对象。
- `__init__()` 的其他参数：初始化所需数据，取决于类的设计。
- 不同魔法方法的签名不同，不能随意增减参数。

## 返回值

| 方法 | 对应操作 | 返回约定 |
| --- | --- | --- |
| `__init__()` | 初始化实例 | `None` |
| `__str__()` | `str(obj)`、通常的 `print(obj)` | 字符串 |
| `__len__()` | `len(obj)` | 非负整数 |
| `__eq__()` | `a == b` | 本文返回布尔值；不支持该对象类型时返回 `NotImplemented` |

## 示例 1：统一角色的打印格式

```python
class Character:
    def __init__(self, name: str, hp: int) -> None:
        self.name = name
        self.hp = hp

    def __str__(self) -> str:
        return f"{self.name} HP: {self.hp}"

character = Character("喵喵", 80)
print(character)
print(str(character))
```

输出：

```text
喵喵 HP: 80
喵喵 HP: 80
```

`__str__()` 返回字符串，不应只在里面打印而不返回。宝可梦项目将角色显示信息集中放在这个方法中，多个菜单便能复用同一种格式。

## 示例 2：为动物园定义长度含义

```python
class MyZoo:
    def __init__(self, animals: dict[str, int]) -> None:
        self.animals = animals.copy()

    def __len__(self) -> int:
        return sum(self.animals.values())

zoo = MyZoo({"pig": 5, "monkey": 10})
print(len(zoo))
print(len(zoo.animals))
```

输出：

```text
15
2
```

本例约定动物园对象的长度是动物总数，而内部字典的长度是动物种类数。两者含义不同，需要在类的接口中说明清楚。本例输入均为非负数量，实际接收外部数据时还要做校验。

## 示例 3：按动物种类比较相等

```python
from types import NotImplementedType

class MyZoo:
    def __init__(self, animals: dict[str, int]) -> None:
        self.animals = animals.copy()

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, MyZoo):
            return NotImplemented
        return set(self.animals) == set(other.animals)

first = MyZoo({"pig": 5, "monkey": 10})
second = MyZoo({"pig": 1, "monkey": 1})
third = MyZoo({"cat": 2})
print(first == second)
print(first == third)
print(first == 1)
print(first is second)
```

输出：

```text
True
False
False
False
```

这里沿用练习中的“动物种类相同就相等”约定，因此数量不同仍可以相等。`set(self.animals)` 取得键集合；若需要同时比较数量，可直接比较两个字典。

遇到其他类型时返回 `NotImplemented`，让 Python 尝试后续比较机制。本例与整数比较最终为 `False`。`is` 比较是否同一对象，不受这个相等规则影响。

## 注意事项

- `NotImplemented` 是返回给比较机制的特殊值，`NotImplementedError` 是异常，二者不同。
- `__len__()` 若返回负数或不符合要求的类型，`len()` 会报错；不能随意返回描述字符串。
- 仅有 `__len__()`、没有自定义 `__bool__()` 时，长度还会影响对象的真假判断。
- `__str__()` 主要面向可读输出，`__repr__()` 主要用于对象表示和调试，它们不是同一个方法。
- 先学习常用协议，再根据业务需要增加特殊方法，不需要把所有方法都实现一遍。
- 类和实例的基础见 [Class 类](<Python3-Class类.md>)。
