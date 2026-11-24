# Python3 列表推导式

> 内容来源：Task 0 矩阵创建与转置，Task 1 存活成员筛选。
> 语义核对：[Python 官方文档：列表推导式](https://docs.python.org/3.13/tutorial/datastructures.html#list-comprehensions)、[嵌套列表推导式](https://docs.python.org/3.13/tutorial/datastructures.html#nested-list-comprehensions)、[序列重复](https://docs.python.org/3.13/library/stdtypes.html#common-sequence-operations)。
> 整理日期：2026-10-21。

## 作用

列表推导式把“遍历数据、按条件筛选、计算新元素”集中表达，适合从已有数据构造新列表。

读取菜单输入、提取角色名称、筛选存活成员、创建矩阵都可以用到它。复杂流程仍可写成普通循环，不需要一律压缩为一行。

## 语法

以下为语法示意，不作为完整程序运行：

```python
[expression for item in iterable]
[expression for item in iterable if condition]
[first if condition else second for item in iterable]
[[expression for col in columns] for row in rows]
```

## 参数

这些是表达式中的组成部分，不是函数调用参数：

- `iterable`：提供原始元素的可迭代对象。
- `item`：本次遍历取得的元素。
- `expression`：计算并加入结果列表的值。
- 后置的 `if condition`：筛选条件，不符合时不生成元素。
- 表达式位置的 `first if condition else second`：决定本次生成哪一个值，不会因此减少元素数量。

## 返回值

返回新列表，会立即遍历输入并收集结果。转换过程中发生的异常仍会抛出，不会自动跳过不合法数据。

新建外层列表不意味着内部对象也被复制；直接把已有角色加入结果时，保存的仍是原角色引用。

## 示例 1：转换字符串，再生成平方列表

```python
raw_values = ["1", "2", "3"]
numbers = [int(value) for value in raw_values]
squares = [number * number for number in numbers]

print(numbers)
print(squares)
print(raw_values)
```

输出：

```text
[1, 2, 3]
[1, 4, 9]
['1', '2', '3']
```

每个输入字符串经过 `int()` 后加入新列表，原列表保留原有字符串。如果包含 `"abc"`，转换仍会报错，推导式不会替代输入检查。

## 示例 2：区分筛选与条件表达式

```python
team = [
    {"name": "喵喵", "hp": 80},
    {"name": "皮卡丘", "hp": 0},
    {"name": "杰尼龟", "hp": 30},
]
available = [member["name"] for member in team if member["hp"] > 0]
labels = [
    member["name"] if member["hp"] > 0 else "已昏厥"
    for member in team
]
print(available)
print(labels)
```

输出：

```text
['喵喵', '杰尼龟']
['喵喵', '已昏厥', '杰尼龟']
```

第一个列表只保留存活成员的名称；第二个为每个原成员生成一项显示文字，所以仍有三项。

宝可梦项目还会判断成员不是当前出战对象。这里用简化字典数据说明筛选，不依赖项目导入或完整菜单。

## 示例 3：创建独立矩阵，并进行转置

```python
shared = [[0] * 3] * 2
matrix = [[0 for _ in range(3)] for _ in range(2)]
shared[0][0] = 9
matrix[0][0] = 9

print(shared)
print(matrix)
transposed = [
    [matrix[row][col] for row in range(2)]
    for col in range(3)
]
print(transposed)
```

输出：

```text
[[9, 0, 0], [9, 0, 0]]
[[9, 0, 0], [0, 0, 0]]
[[9, 0], [0, 0], [0, 0]]
```

`shared` 的两行指向同一个内部列表，所以修改第一行也会出现在第二行。`matrix` 的外层推导式每轮创建一个新列表，各行独立。

转置时外层先选一列，内层依次取各行的该列元素，原来的两行三列变为三行两列。这个示例假定矩阵是规则矩形，实际处理不等长行时需另外考虑边界。

## 注意事项

- 多个 `for` 子句的顺序，与展开后的普通嵌套循环顺序一致。
- 筛选用的 `if` 在 `for` 之后；条件表达式的 `if/else` 在生成值的位置，二者含义不同。
- 不建议用 `[print(x) for x in values]` 只执行副作用，这会额外创建装满 `None` 的列表。
- 不要把“新列表”等同于“成员对象全部独立”，详见 [赋值与浅深复制](<Python3-赋值与浅深复制.md>)。
- 列表推导式用方括号；生成器表达式通常用圆括号，后者按需产生值，见 [generator 与 yield](<Python3-generator生成器与yield.md>)。
