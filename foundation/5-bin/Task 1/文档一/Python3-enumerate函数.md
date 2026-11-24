# Python3 `enumerate()` 函数

> 语义核对：[Python 官方文档：enumerate()](https://docs.python.org/3/library/functions.html#enumerate)
> 整理日期：2026-10-18

## 作用

`enumerate()` 在遍历时，为每个元素配上一个依次递增的编号。循环中可以同时取得**编号和元素**，不必自己维护计数器。

## 语法

```python
enumerate(iterable, start=0)
```

## 参数

- `iterable`：可迭代对象，例如列表、元组、字符串或迭代器。
- `start`：起始编号，为整数，默认是 `0`；每取出一个元素，编号递增 `1`。

## 返回值

返回一个 `enumerate` 迭代器，每次迭代产生一个 `(编号, 元素)` 元组。可用 `list()` 查看全部结果。

```python
skills = ["种子炸弹", "寄生种子"]
print(list(enumerate(skills)))
# [(0, '种子炸弹'), (1, '寄生种子')]
```

## 示例 1：默认从 0 开始

```python
skills = ["种子炸弹", "寄生种子"]

for i, skill in enumerate(skills):
    print(f"{i}: {skill}")
```

输出：

```text
0: 种子炸弹
1: 寄生种子
```

每次循环将元组拆分给两个变量：`i` 接收编号，`skill` 接收对应元素，这叫作**解包**。

## 示例 2：从 1 开始生成技能菜单

```python
skills = ["种子炸弹", "寄生种子"]

for i, skill in enumerate(skills, start=1):
    print(f"{i}: {skill}")
```

输出：

```text
1: 种子炸弹
2: 寄生种子
```

`enumerate(skills, 1)` 与 `enumerate(skills, start=1)` 含义相同。

## 示例 3：字符串也可以编号

```python
for i, char in enumerate("cat"):
    print(i, char)
```

输出：

```text
0 c
1 a
2 t
```

## 注意：编号不等于列表索引

`start=1` 只改变生成的编号，不会跳过第一个元素，也不会改变原列表的索引。

| 技能 | 菜单编号（从 1 开始） | 列表索引 |
| --- | --- | --- |
| 种子炸弹 | 1 | 0 |
| 寄生种子 | 2 | 1 |

因此，菜单编号合法时，可通过 `skills[int(choice) - 1]` 取得对应技能；实际处理输入时，先检查输入是否为合法编号。

## 迭代器使用提醒

同一个迭代器消耗完后不会自动从头开始；需要再次遍历时，重新调用 `enumerate()`。

```python
numbered = enumerate(["种子炸弹", "寄生种子"], start=1)
print(list(numbered))
# [(1, '种子炸弹'), (2, '寄生种子')]
print(list(numbered))
# []
```
