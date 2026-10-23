# Python3 输入校验与异常处理

> 内容来源：宝可梦选队、出战和技能菜单的输入问题及修正过程。
> 语义核对：[str.isdigit()](https://docs.python.org/3.13/library/stdtypes.html#str.isdigit)、[int()](https://docs.python.org/3.13/library/functions.html#int)、[异常处理](https://docs.python.org/3.13/tutorial/errors.html#handling-exceptions)。
> 整理日期：2026-10-20。

## 作用

把用户输入的文本转成合法菜单编号；非法输入只提示并重试，不让游戏崩溃，也不消耗一次行动。

`input()` 得到字符串，要分别回答三个问题：字符格式是否符合当前菜单规则、能否转换成整数、整数是否处于合法范围。

## 相关语法、参数与返回值

| 写法 | 输入与作用 | 返回结果 |
| --- | --- | --- |
| `input(prompt)` | 显示提示并读取一行输入 | 字符串 |
| `text.isdigit()` | 检查非空字符串中的字符是否都属于其认可的数字字符 | `bool` |
| `int(text)` | 把可接受的文本转换成整数 | `int`；不能转换时可抛 `ValueError` |
| `valid_choice(choice, max_choice)` | 本项目自定义编号校验 | 合法为 `True`，否则为 `False` |

项目函数的 `choice` 为字符串，`max_choice` 为允许的最大编号，无默认值。函数不选择角色、不执行技能，也不修改原输入。

异常捕获的基本形式：

```python
try:
    # 可能失败的操作
    ...
except ValueError:
    # 处理值无法转换的情况
    ...
```

这是结构示意。`try` 中发生匹配的异常后，会跳到 `except`；没有异常时，跳过该处理分支，继续执行后面的代码。

## 示例 1：isdigit 为 True，不代表 int 一定成功

```python
for text in ["12", "²", "①", "abc"]:
    print(repr(text), text.isdigit())
```

输出：

```text
'12' True
'²' True
'①' True
'abc' False
```

`int("²")`、`int("①")` 会抛 `ValueError`。旧代码先检查 `isdigit()` 就直接转换，因此输入这些字符时程序退出。

某些版本和配置还会限制十进制字符串转整数的长度；本机已复现 5000 位数字触发限制。不能把“字符串只包含数字”理解成“后续转换永远不会失败”。

## 示例 2：先检查、再转换、再判断范围

下面是项目已经完成的公共校验方法，也可作为独立示例运行：

```python
def valid_choice(choice: str, max_choice: int) -> bool:
    if not choice.isdigit():
        return False
    try:
        choice_number = int(choice)
    except ValueError:
        return False
    return 1 <= choice_number <= max_choice

for text in ["1", "5", "6", "0", "abc", "²", "①", " 1 "]:
    print(repr(text), valid_choice(text, 5))
print("5000位数字", valid_choice("1" * 5000, 5))
```

输出：

```text
'1' True
'5' True
'6' False
'0' False
'abc' False
'²' False
'①' False
' 1 ' False
5000位数字 False
```

当前菜单保留原有规则，不接受两侧空白；如果将来希望支持空白，需要明确加入 `strip()`，不能在整理笔记时悄悄改变规则。

长度限制可能因 Python 版本或配置而异。上面超长输入最终都应被拒绝：转换失败时返回 `False`；若环境允许转换，数值也超过编号上限。

## 三个菜单复用同一份校验

```text
选队菜单 → valid_choice()
出战/换人菜单 → valid_choice()
技能菜单 → valid_choice()
```

修正公共函数后，三个入口都能受益，不必各自再复制一份 `try/except`。

校验通过后，编号还要转换成列表索引：菜单编号 1 对应列表索引 0。选队时每选走一只就 `pop()`，剩余候选编号随列表重新排列。

## 无效输入不能推进战斗

技能选择的流程是：

```text
检查本次能否行动（一次）
  → 显示菜单、读取输入
  → 无效：仍在输入循环里重试
  → 有效：执行一次技能，退出输入循环
```

- 麻痹判断放在输入重试循环外；否则重复输入可能错误地多次消耗状态。
- 无效输入不调用 `use_skill()`，不会取消已有蓄力。
- 只有最终合法选择才执行技能，不能因输入无效就让电脑先行动。
- 选择出战成员时，即使编号合法，还要检查该角色是否仍存活。

## 注意事项

| 容易误解的地方 | 本次学到的处理方式 |
| --- | --- |
| `isdigit()` 就能保证转换成功 | 对真正的 `int()` 转换保留异常保护 |
| 只要能转整数就是合法编号 | 还要检查 `1 <= 编号 <= 上限` |
| 把 `isdigit()` 换成 `isdecimal()` 就解决所有问题 | 超长数字仍可能转换失败 |
| 捕获所有异常最省事 | 针对已知转换问题捕获 `ValueError`，不要掩盖其他编程错误 |
| `range` 可以随便当变量名 | 合法但会遮蔽内置函数；本项目改用 `max_choice` |
| 全角数字都应当报错 | 本项目的 `１` 等十进制数字可以正常转换，保留既有行为 |

`EOFError`、用户中断等是另外的交互问题，本次没有新增退出处理；不能把捕获 `ValueError` 写成“所有输入异常都已处理”。

## 检查记录与关联笔记

本次学习过程中，已经验证三个真实菜单面对 `²`、`①`、5000 位数字会继续等待合法输入，随后完成选队、出战或一次技能；无效输入不重复消耗行动。

本文两个标有“输出”的独立示例保存后已逐段执行，实际输出与文中一致；这不代表重新运行了一整局游戏。

- [原有 input 输入笔记](<input输入.md>)
- [类型提示与空值检查](<Python3-类型提示与空值检查.md>)
- [enumerate() 与菜单索引](<Python3-enumerate函数.md>)
