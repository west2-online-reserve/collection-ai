# 校园IF恋（Campus_IF_Love）开发文档

## 一、项目简介

本项目是一个文字校园恋爱交互式小说（IF）。根据作业要求：
- 将代码拆分为至少三个不同的 `.py` 文件：`story.py`、`character.py`、`game.py`、`main.py`
- 全部代码使用 Type Hint 类型注释
- 完成 Bonus 功能：**存档、读档、即时自动存档**，存档使用 JSON 格式，记录剧情块编号、角色对话索引、好感度等状态

### 文件结构
```
campus_if_love/
├── story.py      # 存放所有剧情文本、对话、礼物好感配置数据
├── character.py  # Character 角色类：好感、对话索引、送礼逻辑（Type Hint）
├── game.py       # Game 主游戏逻辑：场景分支、主循环、存档读档（JSON）
└── main.py       # 程序入口，只负责启动游戏
```

### 运行方式
```bash
python main.py
```

---

## 二、开发过程中遇到的问题与解决方案

### 问题 1：礼物「奇怪的石头」读取好感时报错

**现象**：`GIFT_EFFECTS["奇怪的石头"]` 只有 `default` 键，没有各角色名字作为 key。若直接使用 `gift_data[self.name]` 取值，会触发 `KeyError`，程序崩溃。

**原因**：字典用方括号访问，key 不存在会抛出异常。

**解决**：改用字典 `.get()` 安全取值，并增加分支判断——礼物配置若存在 `default` 字段，直接使用 default 值；否则使用 `gift_data.get(self.name, 0)`，找不到角色则返回 0，避免异常。

```python
gift_data = gift_effects.get(gift, {})
if "default" in gift_data:
    val = gift_data["default"]
else:
    val = gift_data.get(self.name, 0)
```

### 问题 2：玩家输入非法选项时程序直接闪退

**现象**：输入 A/B 以外的选项，或输入不存在的礼物名称，代码raise直接抛出异常，游戏终止。

**解决**：删除主动抛出异常的代码，改用 `while` 循环持续接收输入，直到玩家输入合法内容才退出输入循环，提升鲁棒性。

```python
opt = input("请输入选项A/B：").strip()
while opt not in ("A", "B"):
    opt = input("输入只能是A/B，请重新输入：").strip()
```

### 问题 3：Python 对象无法直接保存到 JSON 文件（存档功能难点）

**现象**：`Character` 实例是 Python 对象，`json` 模块不能直接序列化对象写入文件。

**解决思路（对应作业提示的"剧情块编码 + 剧情句编码"方案）**：
1. 在 `Character` 类增加 `to_save_dict()` 方法：把角色实例需要保存的属性导出为普通字典（字典可被 JSON 序列化）。保存内容包括角色名称、角色描述、好感度 `affinity`、对话索引 `dialog_idx`。
2. 增加静态方法 `from_save_dict`（用 `@staticmethod` 装饰）：接收存档字典，新建并返回 `Character` 实例，完成读档重建对象。
3. 存档 JSON 额外记录 `scene_code`（剧情块编码，区分三条角色剧情线），满足作业提示。

```python
def to_save_dict(self) -> Dict:
    return {
        "name": self.name,
        "role": self.role,
        "affinity": self.affinity,
        "dialog_idx": self.dialog_idx
    }

@staticmethod
def from_save_dict(data: Dict):
    char = Character(name=data["name"], role=data["role"], affinity=data["affinity"])
    char.dialog_idx = data["dialog_idx"]
    return char
```
---

## 三、相关问题的理解（代码走查记录）

### 1. `from typing import Dict, List` 是做什么的？

这是 **Python 类型注解（Type Hint）** 的导入语句。Python 3.9 之前，`Dict`、`List` 等泛型类型不能直接写，需要从标准库 `typing` 导入。

```python
from typing import Dict, List

DIALOGUES: Dict[str, List[Dict[str, str]]] = { ... }
```

含义：`DIALOGUES` 是一个字典，key 是字符串，value 是**列表**，列表内每一项又是一个**字典**（字典的 key 和 value 都是字符串）。

- `typing` 里的 `Dict` / `List` 是**给 IDE、类型检查器（如 mypy）看的，不是程序运行必需**
- 运行时 Python 会忽略这些注解，但**作业要求 Type Hint，必须写**
- 好处：满足作业要求、IDE 自动提示、读代码时一眼看懂变量结构

> 补充：Python ≥ 3.9 支持原生写法 `dict[str, list[dict[str, str]]]`，不用导入；但 `from typing import Dict, List` 写法兼容性更好，常用于教学。

### 2. `@staticmethod` 为什么加？

`@staticmethod` 标记这是一个**静态方法**：调用它**不需要先创建对象实例**，它属于类本身。

对比三种方法：
| 类型 | 参数 | 调用方式 | 能否访问实例属性 |
|------|------|----------|------------------|
| 实例方法 | `self` | `obj.method()` | ✅ 可以 |
| 类方法 | `cls` | `Class.method()` | 只能通过 cls 间接访问 |
| 静态方法 | 无 | `Class.method()` | ❌ 不能 |

`from_save_dict` 的任务是**根据存档字典新建一个 Character 对象**——此时我们**还没有角色实例**，所以不能用普通实例方法（实例方法要求必须先有实例）。用 `@staticmethod` 把它封装在 Character 类内部，逻辑更内聚，调用方式：

```python
char = Character.from_save_dict(save_data)
```

### 3. `.get(self.name, 0)` 的用法

`.get()` 是字典的安全取值方法：
```python
dict.get(key, default=None)
```
- 找到 key → 返回对应值
- 找不到 key → 返回默认值（default），**不会报错**

与方括号 `dict[key]` 的区别：方括号找不到 key 会直接抛 `KeyError` 崩溃；`.get()` 返回默认值，安全。

```python
gift_data = {"学姐": 10, "小白": 15, "姐姐": 20}
gift_data.get("学姐", 0)   # 存在，返回 10
gift_data.get("小黑", 0)   # 不存在，返回默认值 0
gift_data["小黑"]          # ❌ KeyError，直接报错
```

### 4. `json.dump(..., ensure_ascii=False, indent=2)` 参数含义

`json.dump()` 作用：把 Python 对象写入文件，生成 JSON 存档文件。
```python
json.dump(save_data, f, ensure_ascii=False, indent=2)
```
- `save_data`：要保存的数据（字典，存游戏状态：好感度、剧情编号、角色信息）
- `f`：打开的文件句柄

**`ensure_ascii=False`**
默认 `ensure_ascii=True`，中文会被转成 `\uXXXX` 编码（如 `"李晓雪"` → `"\u674e\u5c0f\u96ea"`）。设为 `False` 后，JSON 文件里**直接显示中文**，方便阅读和调试存档内容。

**`indent=2`**
控制格式化缩进。默认 `indent=None` 会压缩成一行，难以阅读；`indent=2` 让每一层缩进 2 个空格并自动换行，输出格式化、易读的存档文件。

```json
{
  "scene_code": 1,
  "affinity": 10
}
```

> 小提醒：程序读档时，无论有没有缩进、中文是否转义，`json.load` 都能正常解析。这两个参数只影响文件的可读性，不影响读档逻辑。

---

## 四、JSON 模块常见操作与踩坑（存档功能）

Python 使用内置模块 `json`，无需额外安装。核心四个函数：`dump` / `dumps` / `load` / `loads`。

### 1. 四个核心函数
| 函数 | 作用 | 游戏场景 |
|------|------|----------|
| `json.dump(obj, fp)` | Python 对象 → 写入文件 | 存档 |
| `json.dumps(obj)` | Python 对象 → 字符串 | 打印调试 |
| `json.load(fp)` | 从文件读取 JSON → Python 对象 | 读档 |
| `json.loads(str)` | JSON 字符串 → Python 对象 | 解析 |

**存档（dump）示例：**
```python
game_data = {
    "scene_code": 1,
    "affinity": 15,
    "player_name": "主角",
    "dialog_idx": 0
}
with open("auto_save.json", "w", encoding="utf-8") as f:
    json.dump(game_data, f, ensure_ascii=False, indent=2)
```

**读档（load）示例：**
```python
with open("auto_save.json", "r", encoding="utf-8") as f:
    load_data = json.load(f)
print(load_data["affinity"])
```

### 2. 支持的数据类型映射
✅ 可以直接序列化：`dict`、`list`、`str`、`int`、`float`、`True/False`、`None`
❌ **不能直接序列化自定义类实例**（如 `Character` 对象）

> 解决方案：写 `to_save_dict()` 方法，把角色对象转成普通字典再存档；读档时用字典重新构造对象——这就是作业提示里"把对象转成基础类型再存入 JSON"的实现。

```python
class Character:
    def __init__(self, name: str, affinity: int):
        self.name = name
        self.affinity = affinity

    def to_save_dict(self):
        return {"name": self.name, "affinity": self.affinity}

char = Character("夏同学", 20)
save_dict = char.to_save_dict()  # 可被 json.dump 序列化
```

### 3. 常用参数
1. `ensure_ascii=False`：中文不转义成 `\uXXXX`，存档文件直接显示中文
2. `indent=2`：格式化缩进，方便人阅读
3. `sort_keys=True`（可选）：字典按键排序，一般游戏存档不用

### 4. 常见踩坑
1. **文件编码**：读写必须加 `encoding="utf-8"`，否则中文乱码
```python
with open("save.json", "w", encoding="utf-8") as f:   # ✅
```
2. **自定义对象直接 dump 报错**：`TypeError: Object of type Character is not JSON serializable` → 用 `to_save_dict()` 转换
3. **JSON 语法**：只能用双引号（不能用单引号）；不允许在 JSON 文件里写注释
4. **读档前判断文件是否存在**：否则抛 `FileNotFoundError`
```python
import os
if os.path.exists("auto_save.json"):
    # 执行读档
```

### 5. 游戏存档完整最小示例
```python
import json
import os

def save_game(scene_code: int, affinity: int) -> None:
    data = {"scene_code": scene_code, "affinity": affinity}
    with open("auto_save.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_game():
    if not os.path.exists("auto_save.json"):
        return None
    with open("auto_save.json", "r", encoding="utf-8") as f:
        return json.load(f)

save_game(scene_code=2, affinity=25)
res = load_game()
print(res)
```

---

## 六、字符串方法 `", ".join()` 使用说明

### 1. 作用
`join()` 是字符串对象的**方法**，接收字符串类型的可迭代对象（列表、生成器等），将序列内所有元素**拼接为单个字符串**，调用者（前面的字符串）作为**分隔符**。`", ".join(xxx)` 代表元素之间使用 `, `（逗号+空格）分隔。

```python
role_names = ["李晓雪", "苏沐", "林晚"]
output = ", ".join(role_names)
print(output)   # 输出：李晓雪, 苏沐, 林晚
```

### 2. 常见踩坑：元素必须是字符串
序列内部元素必须全部为字符串。若存放数字，会抛出 `TypeError`，需要先用 `str()` 转换。

```python
ids = [1, 2, 3]
# 错误写法：", ".join(ids)   # TypeError: expected str instance, int found
# 正确写法：
output = ", ".join(str(i) for i in ids)   # 输出：1, 2, 3
```

### 3. 项目场景
- 打印可选角色列表
- 输出多条提示文本、展示存档槽列表
- 相比多次 `+` 字符串拼接，`join` 在拼接大量文本时性能更优
