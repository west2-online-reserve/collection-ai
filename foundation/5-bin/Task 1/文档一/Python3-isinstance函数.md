# Python3 `isinstance()` 函数

> 语义核对：[Python 官方文档：isinstance()](https://docs.python.org/3/library/functions.html#isinstance)
> 整理日期：2026-10-18

## 作用

`isinstance()` 判断对象是否属于指定类型，也认可继承关系，常用于分支判断和参数检查。

## 语法

```python
isinstance(object, classinfo)
```

## 参数

- `object`：要检查的对象。
- `classinfo`：要判断的类型，例如 `int`、`str` 或自定义类；也可以传类型元组，例如 `(int, float)`。两个参数均无默认值。

## 返回值

符合指定类型或其子类时返回 `True`，否则返回 `False`。判断本身不会修改或转换对象。

## 示例 1：检查常见类型

```python
print(isinstance(10, int))
print(isinstance(10.5, int))
print(isinstance("你好", str))
print(isinstance([1, 2], list))
```

输出：

```text
True
False
True
True
```

## 示例 2：接受多种类型

```python
print(isinstance(12, (int, float)))
print(isinstance(12.8, (int, float)))
print(isinstance("12", (int, float)))
```

输出：

```text
True
True
False
```

类型元组表示“满足其中任意一种”，不是必须同时满足。

## 示例 3：作业中的伤害检查

```python
damage = 12.8

if not isinstance(damage, int):
    damage = int(damage)

print(damage)
```

输出：

```text
12
```

`not` 将结果取反，条件表示“damage 不是整数”。真正执行转换的是 `int()`；浮点数转整数会向零截断，不是四舍五入。

## 示例 4：认可继承关系

```python
class Animal:
    pass

class Cat(Animal):
    pass

cat = Cat()
print(isinstance(cat, Cat))
print(isinstance(cat, Animal))
print(type(cat) is Animal)
```

输出：

```text
True
True
False
```

`cat` 的直接类型是 `Cat`，但因为 `Cat` 继承 `Animal`，它也是 `Animal` 的实例。作业中的妙蛙种子与 `Pokemon` 同样存在继承关系。

## 注意事项

| 易混点 | 正确理解 |
| --- | --- |
| `int` 与 `"int"` | 第二个参数应传类型 `int`，不能用类型名称字符串替代 |
| `isinstance("12", int)` | 返回 `False`，不会自动转换字符串 |
| `not isinstance(x, int)` | 表示 x 不是整数；不能据此保证 `int(x)` 一定成功 |
| `isinstance(x, Animal)` | 认可子类实例 |
| `type(x) is Animal` | 只检查直接类型是否恰好为 Animal |

例如，`int("abc")` 会引发 `ValueError`，所以是否需要异常处理应由输入约定决定。

另一个容易忽略的继承关系是：`bool` 是 `int` 的子类。

```python
print(isinstance(True, int))
print(type(True) is int)
```

输出：

```text
True
False
```

因此，若某个输入要求整数并且明确排除布尔值，不能只依赖 `isinstance(value, int)`。
