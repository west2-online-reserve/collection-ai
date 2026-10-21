# Python3 List（列表）

> 内容来源：Task 0 发牌练习、Task 1 宝可梦队伍与状态列表。
> 语义核对：[Python 官方文档：列表](https://docs.python.org/3.13/tutorial/datastructures.html#more-on-lists)、[复制操作](https://docs.python.org/3.13/library/copy.html)。
> 整理日期：2026-10-21。

## 作用

列表按顺序保存一组元素，允许重复，也允许增加、删除和替换元素。扑克牌、玩家队伍、技能菜单都需要按顺序访问，适合使用列表。

列表保存的是对象引用；复制外层列表，不一定复制里面的字典或角色对象。

## 语法

以下是语法示意，各行分别说明一种操作，不作为完整程序运行：

```python
items = [value1, value2]
items[index]
items[start:stop:step]
items.append(value)
items.extend(iterable)
items.pop(index)
items.remove(value)
items.copy()
```

## 参数

- `index`：元素位置，下标从 `0` 开始；`-1` 表示最后一个元素。
- `start`、`stop`、`step`：切片的起点、终点和步长，不包含终点；步长不能为 `0`。
- `value`：要加入、替换或删除的元素。
- `iterable`：提供一批元素的可迭代对象，例如另一个列表。
- `pop()` 不传位置时默认删除最后一项；`remove()` 删除首个与指定值相等的元素。

## 返回值

| 操作 | 返回值 | 是否修改原列表 |
| --- | --- | --- |
| `items[index]` | 对应位置的元素 | 否 |
| `items[start:stop]` | 新列表，属于浅复制 | 否 |
| `append()`、`extend()`、`remove()` | `None` | 是 |
| `pop()` | 被删除的元素 | 是 |
| `copy()` | 新的外层列表，内部元素仍共享 | 否 |

## 示例 1：添加、切片和按位置取走成员

```python
team = ["妙蛙种子", "皮卡丘"]
team.append("杰尼龟")
team.extend(["小火龙", "喵喵"])

print(team[:3])
print(team[-1])
chosen = team.pop(1)
team.remove("小火龙")
print(chosen)
print(team)
```

输出：

```text
['妙蛙种子', '皮卡丘', '杰尼龟']
喵喵
皮卡丘
['妙蛙种子', '杰尼龟', '喵喵']
```

`pop(1)` 取走下标为 `1` 的皮卡丘，剩余元素重新排列。菜单编号从 `1` 开始时，合法编号要减 `1` 才能作为列表下标。

`append(["小火龙", "喵喵"])` 会把整个内部列表当作一个元素；`extend()` 则逐个添加成员。具体区别见 [list.extend() 笔记](<Python3-list.extend方法.md>)。

## 示例 2：赋值、浅复制与深复制

```python
from copy import deepcopy

original = [{"name": "喵喵", "hp": 80}]
alias = original
shallow = original.copy()
deep = deepcopy(original)

shallow[0]["hp"] = 20
print(alias is original)
print(shallow is original)
print(original[0]["hp"], shallow[0]["hp"], deep[0]["hp"])
```

输出：

```text
True
False
20 20 80
```

`alias` 仍指向原列表。`shallow` 的外层列表独立，但第一个字典与原列表共享。这里 `deepcopy()` 还复制了内部字典，所以修改浅复制中的 HP 不会影响深复制结果。

宝可梦组队需要双方各自拥有独立的角色、技能和状态，现有项目分别深复制候选列表。更多边界见 [赋值与浅深复制](<Python3-赋值与浅深复制.md>)。

## 示例 3：遍历副本，删除原列表中的到期项

```python
durations = [0, 0, 2]

for duration in durations[:]:
    if duration <= 0:
        durations.remove(duration)

print(durations)
```

输出：

```text
[2]
```

直接遍历原列表时删除元素，后面的元素会前移，可能漏掉相邻项。这里遍历切片副本，删除发生在原列表中，两项 `0` 都能处理。

宝可梦项目用 `self.statuses[:]` 遍历状态副本，再从原列表移除到期效果。浅复制保留了效果对象引用，递减对象的持续次数仍会影响原对象。

## 注意事项

- 下标越界会抛出 `IndexError`；切片超出边界通常会截取到实际可用部分。
- 空列表执行 `pop()` 会抛出 `IndexError`，`remove()` 找不到目标会抛出 `ValueError`。
- `sort()` 原地排序并返回 `None`；`sorted()` 返回新的排序结果，不能混淆。
- 不要写 `team = team.append(member)`，否则 `team` 会被赋为 `None`。
- 需要独立修改的二维列表不能简单用 `[[0] * 3] * 2` 创建，详见 [列表推导式](<Python3-列表推导式.md>)。
