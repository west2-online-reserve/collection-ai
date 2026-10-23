# Python3 generator（生成器）与 yield

> 内容来源：进阶发牌中传给 `join()` 的生成器表达式；本次补充 `yield` 的独立学习示例。
> 语义核对：[Python 官方文档：生成器](https://docs.python.org/3.13/tutorial/classes.html#generators)、[yield 表达式](https://docs.python.org/3.13/reference/expressions.html#yield-expressions)、[next()](https://docs.python.org/3.13/library/functions.html#next)。
> 整理日期：2026-10-21。

## 作用

生成器按需逐项产生数据，不必预先把所有结果保存为列表。适合顺序处理数据，或把一批值交给其他函数消费。

生成器函数通过 `yield` 交出一项并暂停，下一次请求时继续执行；生成器表达式则用类似推导式的写法提供逐项计算结果。

## 语法

以下为语法示意，不作为完整程序运行：

```python
def generate(limit):
    for item in range(limit):
        yield item

iterator = generate(limit)
next(iterator)
next(iterator, default)
expression_iterator = (expression for item in iterable if condition)
```

## 参数

- `limit`：示例生成器函数的普通参数，控制最多产生多少项。
- `yield value`：把 `value` 作为本次迭代值交给调用方，并暂停执行。
- `next()` 的第一个参数：需要取下一项的迭代器。
- `default`：迭代器耗尽时由 `next()` 返回的默认值；不传则抛出 `StopIteration`。
- 生成器表达式中的 `iterable`、`expression`、`condition`：分别表示输入、计算规则和筛选条件。

## 返回值

调用生成器函数先得到生成器对象，而不是立即得到全部生成值；函数体在请求下一项时执行。生成器对象属于迭代器，可以传给 `next()`、`for`、`list()` 等。

示例中 `Iterator[int]` 表示调用方可以依次取出整数，实际返回的对象仍是生成器。`yield` 交出的项与函数最终结束是两回事。

## 示例 1：观察创建、执行与暂停

```python
from collections.abc import Iterator

def round_numbers(limit: int) -> Iterator[int]:
    for number in range(1, limit + 1):
        print(f"准备产生 {number}")
        yield number

rounds = round_numbers(2)
print("生成器已创建")
print(next(rounds))
print(next(rounds))
```

输出：

```text
生成器已创建
准备产生 1
1
准备产生 2
2
```

创建对象时没有打印“准备产生”。第一次 `next()` 才进入函数，执行到 `yield` 时交出 `1` 并暂停；第二次从暂停处继续，循环进入下一项，再交出 `2`。

暂停时保留局部变量和执行位置，不是每次从函数开头重新运行。最后一次 `yield` 之后还要继续执行，才能确认函数已结束。

## 示例 2：消耗、耗尽与重新创建

```python
from collections.abc import Iterator

def round_numbers(limit: int) -> Iterator[int]:
    current = 1
    while current <= limit:
        yield current
        current += 1

rounds = round_numbers(3)
print(next(rounds))
print(list(rounds))
print(next(rounds, "已结束"))
print(list(rounds))
print(list(round_numbers(3)))
```

输出：

```text
1
[2, 3]
已结束
[]
[1, 2, 3]
```

`next()` 取走第一项，`list()` 继续收集剩余项。耗尽后，同一个生成器不会自动从头开始；需要重新调用生成器函数，创建新的生成器对象。

不传默认值的 `next()` 会在耗尽时抛出 `StopIteration`，`for` 循环自动识别这一结束信号。`next()` 的默认值只处理耗尽，不会掩盖函数体中的其他异常。

## 示例 3：逐项提供扑克牌名称给 join

```python
cards = [("A", 14), ("10", 10), ("3", 3)]
names = (card[0] for card in cards)

print(next(names))
print(" ".join(names))
print(list(names))
print(" ".join(card[0] for card in cards))
```

输出：

```text
A
10 3
[]
A 10 3
```

发牌代码中的 `" ".join(card[0] for card in player1_cards)` 使用同一种表达式。消费生成器时，每张牌才提供自己的显示名称。

`join(names)` 接收已经消耗一项的生成器，所以只拼接剩余两张牌；最后一行建立新的生成器表达式，才能得到完整结果。

## 注意事项

| 比较点 | 列表 | 生成器 |
| --- | --- | --- |
| 保存结果 | 保存当前所有元素 | 保存执行状态，按需产生元素 |
| 下标与长度 | 支持下标和 `len()` | 通常不支持下标和 `len()` |
| 再次遍历 | 列表保留时可重新遍历 | 同一生成器耗尽后不能重来 |
| 适合场景 | 菜单、队伍、反复访问的数据 | 顺序读取、逐项转换的数据 |

- 普通 `return` 结束当前生成器；`yield` 交出一项后允许以后继续执行。生成器不能靠 `return` 一个列表就让该列表逐项产生。
- 延迟计算可以减少中间结果保存，但消费方仍可能收集全部数据；`list()` 会构造列表，`join()` 最终会构造完整字符串。
- 不要在生成器函数中手动抛出 `StopIteration` 来结束；自然执行结束或使用 `return`。
- 把生成器再赋给另一个变量不会复制迭代进度，两个名字仍可指向同一个生成器。
- 宝可梦队伍需要编号选择、随机抽取和多次访问，继续用列表更合适；本篇没有改写现有战斗循环。
- 立即构造列表的写法见 [列表推导式](<Python3-列表推导式.md>)。
