# Python3 类型提示与空值检查

> 内容来源：宝可梦项目的类型补全与检查过程，包含学习者实现和按要求由 AI 完成的类型整理。
> 语义核对：[Python 3.13 typing](https://docs.python.org/3.13/library/typing.html)、[NoReturn](https://docs.python.org/3.13/library/typing.html#typing.NoReturn)、[TYPE_CHECKING](https://docs.python.org/3.13/library/typing.html#typing.TYPE_CHECKING)。
> 本次补充核对：[Literal 字面量类型](https://docs.python.org/3.13/library/typing.html#typing.Literal)（2026-10-20）。
> 工具参考：[Pyright 配置说明](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)。
> 整理日期：2026-10-20。项目现有写法以 Python 3.10 及以上为基础；以下不使用更高版本才支持的新类型语法。

## 作用

类型提示描述函数接收什么、返回什么，以及属性保存什么数据。阅读代码和静态检查时，可以提前发现字符串与整数混用、列表元素不明确、空对象被当成宝可梦调用等问题。

**注解不会自动转换数据，也不会自动执行运行时校验。** `choice: str` 仍需要 `int(choice)` 才能得到整数；输入是否合法仍要由条件和异常处理决定。

## 基本语法

```python
变量名: 类型 = 初始值

def 函数名(参数名: 类型) -> 返回类型:
    ...
```

上面是语法示意，不是可直接运行的完整程序。实例方法中的 `self` 通常由检查器推断，不必每次手动标注。

## 本次用到的类型

| 写法 | 表达的含义 | 项目例子 |
| --- | --- | --- |
| `str` | 字符串 | `input()` 得到的技能编号 |
| `int` | 整数 | HP、编号上限、剩余持续次数 |
| `bool` | 真或假 | 输入是否合法、是否闪避成功 |
| `Literal["player", "computer"] \| None` | 两个指定字符串之一，或者空值 | 宝可梦所属阵营 |
| `int \| float` | 两种数值类型中的一种 | 伤害或治疗入口 |
| `list[pokemon.Pokemon]` | 列表中保存宝可梦实例 | 候选、玩家队伍、电脑队伍 |
| `list[Skill]` | 列表中保存技能实例 | 每只角色的技能列表 |
| `list[Effect]` | 列表中保存效果实例 | 中毒、烧伤、护盾等状态 |
| `pokemon.Pokemon \| None` | 宝可梦对象或者空值 | 尚未选择首发时的当前成员 |
| `-> pokemon.Pokemon` | 正常返回时返回一只宝可梦 | 选择出战成员 |
| `-> None` | 正常结束，不返回有用结果 | 扣血、治疗、打印菜单 |
| `-> NoReturn` | 不会正常返回调用处 | 退出游戏；始终运行的主循环 |

`list[Skill]` 是类型说明，不是技能对象；真正创建对象仍要调用 `Skill` 的具体子类，例如 `Scratch()`。

## 示例 1：字符串和整数使用不同变量

```python
def doubled(value: int) -> int:
    return value * 2

text: str = "12"
number: int = int(text)
print(text, number, doubled(number))
```

输出：

```text
12 12 24
```

宝可梦菜单中同样保留 `choice: str`，用 `choice_number = int(choice)` 保存转换结果。这样同一个变量不会一会儿表示原始文本，一会儿又表示数字。

这不是说 Python 禁止变量重新绑定为其他类型；它是让已经声明的类型约定与实际赋值一致，方便检查和阅读。

## 实例属性怎样声明

以下是项目中的声明片段，使用时位于 `Play` 类内部，不是独立脚本：

```python
self.player_team: list[pokemon.Pokemon] = []
self.computer_team: list[pokemon.Pokemon] = []
self.current_player_pokemon: pokemon.Pokemon | None = None
self.current_computer_pokemon: pokemon.Pokemon | None = None
```

- 空列表暂时看不出将来放什么，显式声明元素类型能补充这个信息。
- `None` 表示还没选择成员，不应伪造一个角色填上去。
- 父类类型可以描述其子类实例，因此队伍能同时保存妙蛙种子、杰尼龟、小火龙、皮卡丘和喵喵。
- `hp = 80` 这样的简单值常能直接推断为整数，不需要把每个局部变量都重复标注一遍。

## 示例 2：先排除 None，再使用对象

```python
class Item:
    def __init__(self, name: str) -> None:
        self.name = name

def describe(current: Item | None) -> str:
    if current is None:
        return "未选择"
    return current.name

print(describe(None))
print(describe(Item("喵喵")))
```

输出：

```text
未选择
喵喵
```

进入最后一行 `return current.name` 时，前面的空值分支已经返回，剩余路径上的 `current` 就是 `Item`。这叫作类型收窄。

项目的技能入口需要检查双方当前成员。只确认使用者存在，不能保证作为参数传给 `use_skill()` 的对手也存在。

`Pokemon | None` 不会自动帮程序添加检查，仍要写出判断。对空值通常使用 `is None` 或 `is not None`。

## 示例 3：None 与 NoReturn 不同

```python
from typing import NoReturn

def finish() -> NoReturn:
    raise SystemExit(0)

try:
    finish()
except SystemExit as exc:
    print("退出码", exc.code)
```

输出：

```text
退出码 0
```

此处捕获 `SystemExit` 是为了演示，避免示例直接退出；项目的退出方法调用 `sys.exit()`。

| 返回注解 | 调用后会发生什么 |
| --- | --- |
| `-> None` | 可以正常执行完，隐式或显式返回 `None` |
| `-> NoReturn` | 执行路径不会正常返回；例如抛异常、退出，或者始终循环 |

`NoReturn` 不表示“不会抛异常”或“永远不会结束进程”。调用处用 `try/except` 捕获退出异常，也不会让退出函数本身变成正常返回。

## Literal：限制具体取值

`Literal` 是 `typing` 模块提供的字面量类型，用来向类型检查器说明：某个位置允许的是几个**具体值**，而不只是某种宽泛的类型。适合阵营、模式等只有少量固定选项的场景。

项目中的声明片段位于实例初始化方法内部：

```python
from typing import Literal

self.side: Literal["player", "computer"] | None = None
```

| 部分 | 含义 |
| --- | --- |
| `self.side` | 当前宝可梦实例的阵营属性 |
| `Literal["player", "computer"]` | 类型约定允许字符串 `"player"` 或 `"computer"`，大小写与拼写必须一致 |
| `\| None` | 也允许空值，表示尚未分配阵营 |
| `= None` | 创建实例时，把属性的初始值设为 `None` |

`| None` 描述允许的类型，`= None` 则执行初始化赋值，两者职责不同。选队后可把属性赋为 `"player"` 或 `"computer"`；`Literal` 本身不负责分配阵营。

与普通字符串类型比较：

| 注解 | 符合类型约定的值 |
| --- | --- |
| `str \| None` | 任意字符串或 `None`，包括 `"enemy"`、拼错的 `"palyer"` |
| `Literal["player", "computer"] \| None` | 只有 `"player"`、`"computer"` 或 `None` |

### 示例 4：声明阵营并修改属性

```python
from typing import Literal

class Member:
    def __init__(self) -> None:
        self.side: Literal["player", "computer"] | None = None

member = Member()
print(member.side)
member.side = "player"
print(member.side)
member.side = "computer"
print(member.side)
```

输出：

```text
None
player
computer
```

三次赋值都符合声明。若改成 `member.side = "enemy"` 或 `member.side = "palyer"`，静态类型检查器会将其识别为不符合声明的赋值，帮助提前发现错误。

### 示例 5：Literal 不自动执行运行时校验

下面故意违反类型约定，只用于演示 Python 运行时不会自动阻止错误赋值，不是推荐的业务写法：

```python
from typing import Literal

side: Literal["player", "computer"] | None = "enemy"
print(side)
```

普通 Python 执行输出：

```text
enemy
```

静态类型检查应报告该赋值不兼容，但普通 Python 执行仍会得到字符串 `"enemy"`。需要在运行时拒绝非法输入时，仍应显式判断，例如检查值是否属于允许的选项；类型注解不会自动添加这段判断。

## TYPE_CHECKING、前向引用与循环导入

`pokemon.py` 会导入技能，技能又需要在注解中提到宝可梦。仅用于注解的反向导入可以使用下面的模式；这是项目片段，不作为独立运行示例：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemon import Pokemon

def describe_target(target: "Pokemon") -> str:
    return target.name
```

`TYPE_CHECKING` 在正常运行时为 `False`，类型检查器按检查用途处理这个分支。带引号的 `"Pokemon"` 可让注解延后解析，但不会在运行时创建一个名为 `Pokemon` 的对象。

项目也使用了 `from __future__ import annotations` 来延后注解求值；它必须放在文件开头允许的位置。它和 `TYPE_CHECKING` 分别解决注解求值时机与是否执行导入的问题。

主动调用 `typing.get_type_hints()` 时，仍要能找到实际类型名称。不能只因为普通导入成功，就假定所有注解随时都能在缺少名称的环境中解析。

运行时真正需要创建的效果或技能，仍然需要正常导入，不能全部移入 `TYPE_CHECKING`。

## 项目中的严格检查

已保存的 [pyrightconfig.json](<../Pokemon_Example_副本/pyrightconfig.json>)：

```json
{
    "include": ["play.py", "pokemon.py", "skills.py", "effects.py"],
    "typeCheckingMode": "strict"
}
```

在项目目录中，本次实际运行过的命令：

```bash
npm exec --cache /private/tmp/pokemon-typing-npm-cache --yes --package pyright -- pyright --project pyrightconfig.json --outputjson
```

该命令需要本机有 npm，工具包放在临时缓存；缓存缺失时可能需要联网获取。项目没有因此新增 Python 第三方依赖。

2026-10-20 的结果：Pyright 1.1.414 实际分析四个文件，错误、警告和信息均为 0。**这是真实命令行检查结果；没有实际操作编辑器中的 Pylance 面板。**

类型整理前后，还对照了 120 组固定种子的完整三对三，输出、回合数和最终状态一致。运行验证与类型检查各自提供证据，不能相互代替。

## 注意事项

- 不用 `Any` 或忽略诊断来掩盖“可能为 None”的问题，先明确数据有哪些合法状态。
- `Literal` 限定的是类型约定中的具体取值，不是运行时强制约束；不能靠它替代输入校验。
- 父类和子类重写同一方法时，需要保持兼容的参数与返回约定。
- `NoReturn` 只用于不会正常返回的路径，不能当作 `None` 的另一种拼法。
- 类型检查通过，仍可能存在数值、时机、规则或输入问题；功能测试仍然需要。
- 本次四文件类型补全由 AI 按学习者明确要求完成；阅读和复习时仍需要能解释每个类型的含义。

## 示例验证与关联笔记

本文三个标有“输出”的独立示例保存后已逐段执行，实际输出与文中一致；项目片段、配置与命令不属于这三个独立示例。

2026-10-20 补充 `Literal` 后，新增示例 4、5 也已分别执行，实际输出与文中一致；示例 5 故意违反静态类型约定，用于验证运行时不会自动校验。本次只补充笔记与运行这两个短示例，没有重新验收项目或编辑器 Pylance。

- [宝可梦对战阶段学习笔记](<Python3-宝可梦对战阶段学习笔记.md>)
- [输入校验与异常处理](<Python3-输入校验与异常处理.md>)
- [isinstance() 函数](<Python3-isinstance函数.md>)
- [函数参数、单独的 `*` 与实例属性](<Python3-函数参数与实例属性.md>)：区分关键字参数限制、参数默认值和 `self.speed` 的赋值保存。
- [dataclass 数据类与冻结实例](<Python3-dataclass数据类与冻结实例.md>)：解释 `SkillAction` 的字段、自动初始化和冻结限制；类型注解与冻结都不代替业务校验。
