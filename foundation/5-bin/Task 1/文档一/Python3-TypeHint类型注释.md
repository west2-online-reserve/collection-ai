# Python3 Type Hint（类型注释）

> 内容来源：Task 1 方法参数、返回值、角色列表与可空出战成员的类型整理。
> 语义核对：[Python 官方文档：typing](https://docs.python.org/3.13/library/typing.html)、[变量注解](https://docs.python.org/3.13/reference/simple_stmts.html#annotated-assignment-statements)。
> 整理日期：2026-10-21。

## 作用

类型注释描述变量、参数和返回结果期望使用的数据类型，帮助阅读代码，也让编辑器和静态检查工具尽早发现不兼容的调用。

**注释不等于类型转换，也不等于运行时校验。** 输入仍须转换、检查范围，并根据情况捕获异常。

## 语法

以下为语法示意，不作为完整程序运行：

```python
name: str = "喵喵"
team: list[str] = []
counts: dict[str, int] = {}

def calculate(value: int | float) -> int:
    ...

current: str | None = None
```

本篇采用 Python 3.10 或以上可用的 `X | Y` 联合类型写法。`list[str]`、`dict[str, int]` 等内置容器注解从 Python 3.9 起可用。

## 参数

- 参数名后的 `: Type`：声明函数期望接收的数据类型。
- `list[T]`：列表元素的类型约定。
- `dict[K, V]`：字典的键类型和值类型约定。
- `X | Y`：值可以属于其中任意一种类型，不表示自动在二者之间转换。
- `X | None`：值可能是一个 `X`，也可能暂时没有对象。
- `self.attribute: Type`：在实例属性赋值时写出类型；`self` 通常不必重复标注。

## 返回值

`-> Type` 说明函数正常返回结果的类型，实际结果仍由函数体中的 `return` 决定。

| 写法 | 表达的含义 |
| --- | --- |
| `-> int` | 正常返回整数 |
| `-> list[str]` | 正常返回字符串列表 |
| `-> None` | 不返回供调用方使用的结果，常用于修改状态的方法 |
| `-> str | None` | 可能返回字符串，也可能返回 `None` |

注解本身不会创建默认值；只写属性类型声明，不代表属性已经初始化。

## 示例 1：声明列表、字典和返回结果

```python
def count_skills(skills: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for skill in skills:
        counts[skill] = counts.get(skill, 0) + 1
    return counts

skills: list[str] = ["火花", "水枪", "火花"]
result = count_skills(skills)
print(result)
print(result["火花"])
```

输出：

```text
{'火花': 2, '水枪': 1}
2
```

参数约定列表里存放字符串，返回约定字典以字符串为键、整数为值。调用方仍然得到普通 Python 字典，不是另一种特殊的运行时容器。

## 示例 2：排除 None 后访问成员属性

```python
class Character:
    def __init__(self, name: str) -> None:
        self.name: str = name

def show_current(current: Character | None) -> str:
    if current is None:
        return "尚未选择成员"
    return current.name

print(show_current(None))
print(show_current(Character("喵喵")))
```

输出：

```text
尚未选择成员
喵喵
```

提前处理空值后，后续路径中的 `current` 可以按 `Character` 使用，这称为类型收窄。宝可梦项目中的当前出战成员可能为 `None`，调用技能前也需要先确认双方成员存在。

如果只写注解、直接访问 `None.name`，运行时仍会报错，类型声明不会替我们补上分支。

## 示例 3：输入类型声明与真正的转换检查

```python
def valid_choice(choice: str, max_choice: int) -> bool:
    if not choice.isdigit():
        return False
    try:
        choice_number = int(choice)
    except ValueError:
        return False
    return 1 <= choice_number <= max_choice

for value in ["1", "0", "abc", "²"]:
    print(value, valid_choice(value, 3))
```

输出：

```text
1 True
0 False
abc False
² False
```

`choice` 是输入字符串，转换后用另一个整数变量 `choice_number` 保存，避免同一个局部变量先后承担冲突的类型。

`"²".isdigit()` 为真，但 `int("²")` 会抛出 `ValueError`。所以格式检查、转换保护和范围判断各有作用，不能用注解代替。

## 注意事项

- Python 通常不自动强制函数和变量的类型注释；不符合约定的赋值可能继续运行，或在之后的操作中失败。
- `Literal["player", "computer"]` 限制静态类型约定中的具体值，不自动添加运行时白名单检查。
- `-> None` 与 `NoReturn` 不同，后者用于不会正常返回的执行路径。
- 仅用于注解的反向导入可以配合 `TYPE_CHECKING`；运行时实际需要创建的对象仍要正常导入。
- “程序运行过”“类型检查通过”“编辑器 Pylance 诊断已核对”是不同的验证事项，不能互相替代。
- 项目配置、前向引用和严格检查的已有记录见 [类型提示与空值检查](<Python3-类型提示与空值检查.md>)，输入边界见 [输入校验与异常处理](<Python3-输入校验与异常处理.md>)。
