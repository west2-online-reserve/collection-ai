# Python3 re 正则表达式

> 内容来源：Task 0 进阶密码格式校验，补充战斗文本提取与替换示例。
> 语义核对：[Python 官方文档：re 模块](https://docs.python.org/3.13/library/re.html)。
> 整理日期：2026-10-21。

## 作用

正则表达式用一条模式描述文本结构，可用于格式校验、提取信息和替换内容。已有密码练习要检查整段输入是否满足字符与长度限制，因此使用整串匹配。

## 语法

以下为语法示意，不作为完整程序运行：

```python
import re

re.fullmatch(pattern, text)
re.match(pattern, text)
re.search(pattern, text)
re.findall(pattern, text)
re.sub(pattern, replacement, text)
compiled = re.compile(pattern)
```

## 参数

- `pattern`：正则模式，常用原始字符串写法，例如 `r"[0-9]+"`。
- `text`：待处理文本；本篇使用字符串。
- `replacement`：替换内容，本篇使用普通字符串。
- `flags`：可选匹配标志，例如 `re.IGNORECASE` 忽略字母大小写。
- `sub()` 的 `count`：最多替换次数，默认 `0` 表示替换全部匹配项。

| 模式 | 含义 |
| --- | --- |
| `[0-9]` | 一个 ASCII 数字 |
| `[A-Za-z]` | 一个英文字母 |
| `\d` | 默认字符串模式下的 Unicode 十进制数字 |
| `\s` | 空白字符 |
| `+` | 前面的规则重复一次或多次 |
| `{6,18}` | 前面的规则重复 6～18 次 |
| `(规则)` | 捕获分组，可单独提取 |

## 返回值

| 方法 | 匹配范围与结果 |
| --- | --- |
| `fullmatch()` | 整串符合时返回匹配对象，否则 `None` |
| `match()` | 从文本开头匹配，成功返回匹配对象，否则 `None` |
| `search()` | 查找首个匹配位置，成功返回匹配对象，否则 `None` |
| `findall()` | 返回匹配结果列表，具体元素形状受捕获分组影响 |
| `sub()` | 返回替换后的新字符串 |
| `compile()` | 返回可重复使用的正则模式对象 |

## 示例 1：校验 6～18 位数字或英文字母

```python
import re

rule = re.compile(r"[0-9A-Za-z]{6,18}")
for password in ["abc123", "abc", "abc_123", "１２３abc"]:
    print(password, rule.fullmatch(password) is not None)
```

输出：

```text
abc123 True
abc False
abc_123 False
１２３abc False
```

字符集合不包含下划线，也不包含全角数字。原练习中的 `\d` 会接受更多 Unicode 十进制数字；若要求仅半角数字，写 `[0-9]` 更明确。

`fullmatch()` 要求整个密码符合模式。只用 `search()` 找到局部符合的片段，不能证明整串格式正确。格式通过也不代表密码足够安全。

## 示例 2：从日志中提取生命值和数字

```python
import re

text = "喵喵 HP: 60/80"
match = re.search(r"HP: ([0-9]+)/([0-9]+)", text)
if match is not None:
    print(match.group(0))
    print(match.groups())
    print(int(match.group(1)) + 5)

print(re.findall(r"[0-9]+", "回合12，伤害30"))
print(re.search(r"HP", "没有生命值信息") is None)
```

输出：

```text
HP: 60/80
('60', '80')
65
['12', '30']
True
```

`group(0)` 是整段匹配，`group(1)`、`group(2)` 分别是两个捕获分组。提取出来仍是字符串，数值运算需要 `int()` 等转换。

使用 `.group()` 前先排除 `None`。`findall()` 若使用捕获分组，会返回分组内容；有多个捕获分组时，结果元素通常是元组。

## 示例 3：替换空白或数字

```python
import re

text = "  火花   水枪\t护盾  "
cleaned = re.sub(r"\s+", " ", text).strip()
print(cleaned)
print(re.sub(r"[0-9]+", "#", "HP80/100"))
print(re.sub(r"[0-9]+", "#", "HP80/100", count=1))
```

输出：

```text
火花 水枪 护盾
HP#/#
HP#/100
```

连续空白被归并为一个空格，`strip()` 再移除两端空白。替换函数返回新字符串，不会原地改变旧字符串。

## 注意事项

- 原始字符串前缀 `r` 减少 Python 字符串层面的转义干扰，正则引擎仍会解释 `\d`、`\s` 等规则。
- `\w` 不等于只允许英文字母和数字，它还会包含下划线及其他 Unicode 字符。
- 特殊字符需要按模式规则转义；单纯查找固定文本时，也可以使用普通字符串方法。
- 本篇统一使用字符串模式和字符串文本，不混用 `bytes`。
- 菜单编号通常直接做整数转换与范围检查即可，相关边界见 [输入校验与异常处理](<Python3-输入校验与异常处理.md>)。
