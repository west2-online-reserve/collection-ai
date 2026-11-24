# Python3 `list.extend()` 方法

> 内容来源：学习宝可梦组队代码 `self.computer_team.extend(chosen_team)` 时的提问。
> 语义核对：[Python 官方文档：列表详解](https://docs.python.org/zh-cn/3.13/tutorial/datastructures.html#more-on-lists)、[可变序列类型](https://docs.python.org/zh-cn/3.13/library/stdtypes.html#mutable-sequence-types)。
> 整理日期：2026-10-20。

## 作用

`extend()` 把可迭代对象中的元素逐个追加到原列表末尾。已有元素保留，新元素按遍历顺序加入；它不是替换整个列表，也不会自动去重。

## 语法

```python
target_list.extend(iterable)
```

## 参数

- `target_list`：需要添加元素的列表。
- `iterable`：提供元素的可迭代对象，可以是列表、元组、字符串或迭代器等。字符串会逐个提供字符，迭代器则会被遍历消耗。
- `extend()` 接收一个可迭代对象参数，不是多个待添加元素。

## 返回值

**返回 `None`，直接修改原列表，不返回扩展后的列表。**

因此不要写 `self.computer_team = self.computer_team.extend(chosen_team)`，否则属性会被重新绑定为 `None`。

## 示例 1：把选好的成员追加到队伍

```python
computer_team = ["妙蛙种子"]
chosen_team = ["皮卡丘", "杰尼龟", "喵喵"]

result = computer_team.extend(chosen_team)

print(computer_team)
print(len(computer_team))
print(result)
```

输出：

```text
['妙蛙种子', '皮卡丘', '杰尼龟', '喵喵']
4
None
```

原来的妙蛙种子不会被清除，所以队伍已有成员时，再扩展会继续累加。示例中的四名成员用于说明列表行为，不表示项目允许四人队伍。

在提问中的代码里，`self` 是游戏控制器实例，`self.computer_team` 是电脑队伍列表，`chosen_team` 是准备加入的成员集合。

当 `chosen_team` 是另一份普通列表时，可以用下面的循环理解添加过程；这是说明片段，不能紧接原来的 `extend()` 再执行一遍，否则会重复添加：

```python
for member in chosen_team:
    self.computer_team.append(member)
```

## 示例 2：与 `append()` 的区别

```python
chosen_team = ["皮卡丘", "杰尼龟"]

a = []
a.extend(chosen_team)

b = []
b.append(chosen_team)

print(a)
print(len(a))
print(b)
print(len(b))
```

输出：

```text
['皮卡丘', '杰尼龟']
2
[['皮卡丘', '杰尼龟']]
1
```

| 方法 | 添加内容 | 示例结果的列表长度 |
| --- | --- | ---: |
| `extend(chosen_team)` | 把里面的元素逐个添加 | 2 |
| `append(chosen_team)` | 把整个列表作为一个元素添加 | 1 |

组队时需要列表中直接保存每只宝可梦；如果使用 `append(chosen_team)`，得到的会是嵌套列表，后续把每个元素当宝可梦使用时就不符合预期。

## 示例 3：元组、字符串、空列表与重复元素

```python
values = [1]
values.extend((2, 3))
print(values)

letters = []
letters.extend("cat")
print(letters)

values.extend([])
print(values)

values.extend([3])
print(values)
```

输出：

```text
[1, 2, 3]
['c', 'a', 't']
[1, 2, 3]
[1, 2, 3, 3]
```

传入空列表不会新增元素；重复元素会照常加入。想把整个字符串作为一个元素添加时，用 `append("cat")`。

`extend()` 只逐个添加输入中的元素，不会递归展开嵌套列表。

## 示例 4：添加的是对象引用，不是深复制

下面用字典模拟具有血量的角色，便于独立运行：

```python
member = {"name": "皮卡丘", "hp": 80}
chosen_team = [member]
computer_team = []
computer_team.extend(chosen_team)

print(computer_team is chosen_team)
print(computer_team[0] is chosen_team[0])

computer_team[0]["hp"] = 10
print(chosen_team[0]["hp"])

computer_team.append({"name": "杰尼龟", "hp": 80})
print(len(computer_team), len(chosen_team))
```

输出：

```text
False
True
10
2 1
```

外层列表是两份，但第一个位置指向同一个字典。修改成员血量，两处都能看到；只给电脑名单增加一个成员，不会给原名单也增加成员。

宝可梦对象也是如此：`extend()` 不会创建新的宝可梦，队伍是否独立取决于 `chosen_team` 中的对象此前如何创建或复制。不能只看到 `extend()` 就判断存在共享错误，也不能把它当成深复制。

## 注意事项

- 扩展空队伍后，队伍长度等于加入的元素数量；扩展非空队伍会在原人数上累加。
- 它不负责验证队伍人数、成员是否存活或是否重复，这些是游戏逻辑的职责。
- 不要用它的返回值覆盖队伍列表；调用后直接继续使用原列表。
- 需要独立成员时，应检查成员的创建或复制过程，而不是依赖 `extend()`。

## 示例验证与关联笔记

本文四个标有“输出”的独立示例已使用 `python3 -B` 逐段执行，实际输出与文中一致。语法和等价循环片段不属于这四个独立示例。本次只整理笔记，不修改游戏源码、测试或配置。

- [赋值、浅复制与深复制](<Python3-赋值与浅深复制.md>)
- [enumerate() 函数](<Python3-enumerate函数.md>)
