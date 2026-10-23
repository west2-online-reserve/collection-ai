# Python3 Dict（字典）

> 内容来源：Task 0 字典删除与计数练习、校园礼物好感度和宝可梦属性克制表。
> 语义核对：[Python 官方文档：字典](https://docs.python.org/3.13/library/stdtypes.html#mapping-types-dict)。
> 整理日期：2026-10-21。

## 作用

字典保存“键 → 值”的对应关系。通过礼物名称查好感度、通过属性查伤害倍率，比用列表位置表达这些关系更直观。

键不能重复；向已有键赋值会更新对应值。字典保留插入顺序，但不会自动按键大小排序。

## 语法

以下为语法示意，不作为完整程序运行：

```python
data = {key: value}
data[key] = new_value
data.get(key, default)
key in data
data.items()
data.pop(key)
```

## 参数

- `key`：用于查询的键，必须可哈希；字符串、整数通常可以作为键，列表和字典不能直接作为键。
- `value`、`new_value`：键对应的数据，可以是数值、字符串、列表或其他对象。
- `default`：`get()` 找不到键时返回的值，省略时为 `None`。
- `pop(key, default)` 可传入缺失时的默认值；省略默认值且键不存在会抛出 `KeyError`。

## 返回值

| 写法 | 返回内容 |
| --- | --- |
| `data[key]` | 对应值；键不存在时抛出 `KeyError` |
| `data.get(key, default)` | 已有值或默认值，不会自动插入新键 |
| `key in data` | 键是否存在的布尔值 |
| `data.items()` | 随字典变化的键值对视图，可供遍历 |
| `data.pop(key)` | 删除并返回对应值 |

## 示例 1：通过礼物名称查询好感度

```python
gifts = {"咖啡": 5, "书": 8}
gifts["花"] = 6
gifts["咖啡"] = 7

print(gifts["咖啡"])
print(gifts.get("蛋糕", 0))
print("蛋糕" in gifts)
for name, value in gifts.items():
    print(name, value)
```

输出：

```text
7
0
False
咖啡 7
书 8
花 6
```

这组好感度数值仅用于说明字典，不替代校园项目实际数据。更新咖啡的值没有增加新键，也没有改变它原来的插入位置。

`get("蛋糕", 0)` 返回 `0`，随后查键仍为 `False`，说明默认值没有自动写回。

## 示例 2：一次遍历完成计数和属性查表

```python
skills = ["火花", "水枪", "火花"]
counts = {}
for skill in skills:
    counts[skill] = counts.get(skill, 0) + 1
print(counts)

multipliers = {"Fire": {"Grass": 2.0, "Water": 0.5}}
print(multipliers.get("Fire", {}).get("Grass", 1.0))
print(multipliers.get("Normal", {}).get("Fire", 1.0))
```

输出：

```text
{'火花': 2, '水枪': 1}
2.0
1.0
```

计数时，第一次遇到某个技能就从 `0` 加到 `1`，之后在已有次数上增加。

属性表先查攻击方，再查防守方。外层查不到时返回空字典，内层继续以 `1.0` 作为默认倍率；这是项目对未定义属性关系的处理约定。

## 示例 3：先取得键的快照，再删除字典项

```python
students = {"001": "zhangsan", "010": "lisi", "901": "wangwu"}

for key in list(students):
    if int(key) % 2 == 0:
        students.pop(key)

print(students)
```

输出：

```text
{'001': 'zhangsan', '901': 'wangwu'}
```

`list(students)` 取得一份键列表，相当于这里的 `list(students.keys())`。删除发生在原字典中，不会改变用于遍历的列表。

## 注意事项

- 直接遍历字典或其视图时增加、删除键，可能抛出 `RuntimeError` 或使遍历不完整；需要先取快照或构建新字典。
- 只更新已有键的值与改变键集合不同，不能概括成“遍历时任何修改都不允许”。
- `in` 默认查键，不是查值；查值需明确使用 `data.values()`。
- `get()` 的默认值不能替换一个已经存在、值恰好为 `None` 的键。
- 字典的浅复制仍可能共享内部列表或对象，详见 [赋值与浅深复制](<Python3-赋值与浅深复制.md>)。
