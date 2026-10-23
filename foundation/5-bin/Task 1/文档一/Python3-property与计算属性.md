# Python3 `@property` 与计算属性

> 内容来源：学习者对 `@property def display_name(self) -> str` 的提问。
> 语义核对：[Python 官方文档：property()](https://docs.python.org/zh-cn/3.13/library/functions.html#property)。
> 整理日期：2026-10-20。

## 作用

`@property` 让一个读取方法可以通过属性形式访问。调用者写 `obj.display_name`，Python 执行对应的读取方法，把返回结果交给调用者。

适合根据已有数据计算显示名称等结果，避免另外保存一份需要同步更新的数据。`property` 是内置功能，无需额外导入。

## 用法

提问中的代码如下；它需要放在类内部，并确保实例能读取到 `name` 和 `side`，不是独立运行脚本：

```python
@property
def display_name(self) -> str:
    if self.side is None:
        return self.name
    return f"[{self.side}] {self.name}"
```

| 部分 | 含义 |
| --- | --- |
| `@property` | 将下面的方法作为同名属性的读取方法，即 getter |
| `self` | 正在读取属性的当前实例 |
| `display_name` | 对外读取的属性名称 |
| `-> str` | 声明读取结果预期为字符串，不是自动转换或运行时校验 |
| `self.side is None` | 判断是否没有阵营标记，只检查 `None` |
| `return self.name` | 没有阵营时直接返回原始名称，并结束本次读取 |
| `f"[{self.side}] {self.name}"` | 有阵营时拼接显示名称，不修改原始 `name` |

访问时使用 `p.display_name`，不要写成 `p.display_name()`。只有在通过实例读取时，才会执行本例的读取方法；在类上访问 `Character.display_name` 得到的是 property 对象本身。

## 示例 1：根据当前阵营动态生成名称

下面使用独立的简化角色类，不导入或修改宝可梦项目：

```python
class Character:
    def __init__(self, name: str, side: str | None = None) -> None:
        self.name = name
        self.side = side

    @property
    def display_name(self) -> str:
        if self.side is None:
            return self.name
        return f"[{self.side}] {self.name}"

p = Character("喵喵")
print(p.display_name)

p.side = "玩家"
print(p.display_name)

p.side = "电脑"
print(p.display_name)
print(p.name)
```

输出：

```text
喵喵
[玩家] 喵喵
[电脑] 喵喵
喵喵
```

每次读取都会执行这个 getter，使用当时的 `side` 和 `name`，所以修改阵营后显示会变化。原始名称仍是“喵喵”，不会被加上阵营前缀覆盖。

在对战日志中，这种形式可以区分双方的同名角色，例如“[玩家] 喵喵”和“[电脑] 喵喵”。这是对提问片段用途的说明，不替代对项目实际日志路径的检查。

## 示例 2：与初始化时保存一次字符串的区别

```python
class Character:
    def __init__(self, name: str, side: str) -> None:
        self.name = name
        self.side = side
        self.saved_display_name = f"[{self.side}] {self.name}"

    @property
    def display_name(self) -> str:
        return f"[{self.side}] {self.name}"

p = Character("喵喵", "玩家")
p.side = "电脑"

print(p.saved_display_name)
print(p.display_name)
```

输出：

```text
[玩家] 喵喵
[电脑] 喵喵
```

`saved_display_name` 保存的是初始化时得到的字符串，改变 `side` 不会自动重写它。`display_name` 在读取时重新生成结果。

普通 `@property` 不会自动缓存返回值；本例没有使用缓存机制。它也不是监听到 `side` 改变就主动运行，而是在下一次读取 `display_name` 时才计算。

## 示例 3：只读限制、误加括号和空字符串

```python
class Character:
    def __init__(self, name: str, side: str | None = None) -> None:
        self.name = name
        self.side = side

    @property
    def display_name(self) -> str:
        if self.side is None:
            return self.name
        return f"[{self.side}] {self.name}"

p = Character("喵喵", "玩家")

try:
    p.display_name = "新名称"
except AttributeError:
    print("AttributeError：没有定义 display_name 的 setter")

try:
    p.display_name()
except TypeError:
    print("TypeError：不能把返回的字符串当函数调用")

p.name = "皮卡丘"
print(p.display_name)

p.side = ""
print(p.display_name)

p.side = None
print(p.display_name)
```

输出：

```text
AttributeError：没有定义 display_name 的 setter
TypeError：不能把返回的字符串当函数调用
[玩家] 皮卡丘
[] 皮卡丘
皮卡丘
```

本例只定义 getter，没有定义 `@display_name.setter`，因此不能直接赋值给 `display_name`。但 `name` 和 `side` 仍可修改，这不等于冻结整个对象。

`p.display_name()` 会先读取属性得到字符串，再尝试调用这个字符串，所以报 `TypeError`。错误原因取决于 getter 返回的对象，本例返回的是字符串。

空字符串 `""` 不是 `None`，因此仍走拼接分支，产生空的方括号；只有 `None` 会走直接返回名称的分支。

## 普通属性、普通方法与 property 的比较

| 写法 | 访问时发生什么 | 本例中的用途 |
| --- | --- | --- |
| `p.name` | 读取已经保存的普通属性值 | 原始名称 |
| `p.get_display_name()` | 显式调用普通方法，取得结果 | 未使用装饰器时的一种写法 |
| `p.display_name` | 属性访问触发 getter，取得其结果 | 动态显示名称 |

上表中的普通方法名称只是比较用的假设，并不是前面三个示例实际定义的方法。

## 注意事项

- 不要在本例的 `__init__()` 中再写 `self.display_name = ...`；该 property 没有 setter，这种赋值同样会报错。
- `self.name` 直接返回原值，因此应保证它符合字符串约定；`-> str` 不会自动把数字变成字符串。
- `@property` 不会自动创建 `name` 和 `side`，这些底层数据仍需通过初始化、类属性或其他明确路径提供。
- 本例 getter 只计算显示文本，不改变血量、阵营等战斗状态，也不自动执行打印。
- 此处“只读”指不能通过 `display_name` 直接设置值；与 `@dataclass(frozen=True)` 的实例赋值限制范围不同，两者都不应被理解为递归冻结所有引用对象。

## 示例验证与关联笔记

本文三个标有“输出”的独立示例已使用 `python3 -B` 逐段执行，实际输出与文中一致。提问中的类内部片段不属于这三个独立示例；本次只整理笔记，不修改业务代码、测试或项目配置。

- [函数参数、单独的 `*` 与实例属性](<Python3-函数参数与实例属性.md>)
- [dataclass 数据类与冻结实例](<Python3-dataclass数据类与冻结实例.md>)
- [类型提示与空值检查](<Python3-类型提示与空值检查.md>)
