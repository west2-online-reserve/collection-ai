# Python3 Lambda 匿名函数

> 内容来源：Task 0 进阶斗地主发牌中的 `key=lambda x: x[1]`。
> 语义核对：[Python 官方文档：Lambda 表达式](https://docs.python.org/3.13/tutorial/controlflow.html#lambda-expressions)、[排序指南](https://docs.python.org/3.13/howto/sorting.html#key-functions)。
> 整理日期：2026-10-21。

## 作用

Lambda 用一个表达式创建函数对象，适合临时传入简单计算规则。发牌排序需要“从一张牌里取出数值大小”，可以把这个规则直接交给排序函数。

## 语法

以下为语法示意，不作为完整程序运行：

```python
lambda parameter: expression
lambda first, second: expression
sorted(items, key=lambda item: expression, reverse=True)
```

## 参数

- `parameter`、`first`、`second`：匿名函数接收的参数，用法与普通函数参数类似。
- `expression`：计算结果的单个表达式，结果自动作为函数返回值。
- 排序中的 `key`：接收一个函数，排序时用它为每个元素计算比较依据。
- `reverse=True`：排序函数的参数，表示降序；它不是 Lambda 自身的参数。

## 返回值

Lambda 表达式本身产生一个函数对象；调用该函数时，返回冒号右侧表达式的计算结果。它不会自动执行排序，排序仍由 `sorted()` 或列表的 `sort()` 完成。

## 示例 1：把两数相加写成匿名函数

```python
add = lambda a, b: a + b
print(add(3, 4))

def add_with_def(a, b):
    return a + b

print(add_with_def(3, 4))
```

输出：

```text
7
7
```

两种函数在本例完成同一件事。这里给 Lambda 赋名只用于观察调用方式；需要长期复用、有明确名称的函数通常用 `def` 更清楚。

## 示例 2：按扑克牌的数值降序排列

```python
cards = [("3", 3), ("A", 14), ("10", 10)]
ordered = sorted(cards, key=lambda card: card[1], reverse=True)

print([card[0] for card in ordered])
print([card[0] for card in cards])
```

输出：

```text
['A', '10', '3']
['3', 'A', '10']
```

牌面用于显示，第二项用于比较大小。`lambda card: card[1]` 只返回一张牌的排序依据，`sorted()` 才返回排序后的新列表。原列表不变。

## 示例 3：按多个依据排列待行动成员

```python
actions = [
    {"name": "普通攻击", "priority": 0, "speed": 90},
    {"name": "电光一闪", "priority": 1, "speed": 50},
    {"name": "水枪", "priority": 0, "speed": 43},
]
ordered = sorted(
    actions,
    key=lambda action: (action["priority"], action["speed"]),
    reverse=True,
)
for action in ordered:
    print(action["name"])
```

输出：

```text
电光一闪
普通攻击
水枪
```

元组按位置依次比较：先比较技能先手值，相同时比较速度。这里用于说明排序依据，不包含宝可梦项目的换人、同速随机和原行动者校验规则。

## 注意事项

- Lambda 的函数体只能是一个表达式，不能直接写赋值语句、`try/except` 或多行流程。
- 条件表达式可以写入 Lambda，但规则变复杂时应改用 `def`。
- 传给 `key` 的是函数对象，不能提前调用并传入计算结果。
- `list.sort()` 原地排序并返回 `None`，`sorted()` 返回新列表。
- 多个依据都相同时，Python 排序保留原有相对顺序；这不等于随机决定先后。
- 推导式的写法见 [列表推导式](<Python3-列表推导式.md>)。
