# 🛡️ Python 异常处理 (try...except) 笔记



Python 使用 `try...except` 块来管理程序运行时可能出现的错误（称为“异常”）。这能防止程序因意外错误而崩溃，使其更加“健壮”。



## 1. 基本结构：`try...except`（“Plan B”）



这是最核心的用法：尝试执行 `try` 块中的代码 (Plan A)，如果失败，就执行 `except` 块 (Plan B)。

Python

```
try:
    # 1. 尝试执行的代码 (Plan A)
    num = int(input("输入一个数字: "))
    result = 10 / num
    print(f"结果是: {result}")
    
except:
    # 2. 如果 'try' 块中 *任何* 地方出错 (Plan B)
    print("噢！出错了！")

# 3. 无论如何，程序都会继续
print("--- 程序继续 ---")
```

> **⚠️ 警告：** “裸露”的 `except:` (如上) 会捕获 *所有* 错误，包括 `KeyboardInterrupt` (用户按 Ctrl+C)。这通常是坏习惯，因为它会隐藏问题。



## 2. 精确捕获：`except <ErrorType>`



**强烈推荐**的做法是，只捕获你“预料到”的特定异常。

Python

```
try:
    num = int(input("输入一个数字: "))
    result = 10 / num

# “B计划 1”：只在发生 'ValueError' 时触发
except ValueError:
    print("输入无效：你必须输入一个 *数字*！")

# “B计划 2”：只在发生 'ZeroDivisionError' 时触发
except ZeroDivisionError:
    print("输入无效：除数 *不能* 是 0！")
```



### 技巧：



- **捕获多种类型：** 将它们放入一个元组 `()`。

  Python

  ```
  except (ValueError, TypeError):
      print("输入类型或值不正确。")
  ```

- **获取错误信息：** 使用 `as e` 来捕获异常对象，它包含了错误详情。

  Python

  ```
  except ValueError as e:
      print(f"发生了 ValueError，详情: {e}")
      # (输出详情: invalid literal for int() with base 10: 'abc')
  ```

- **捕获所有（但更安全）：** 如果你想捕获所有“常规”错误，请使用 `except Exception as e:`。

  - `Exception` 是几乎所有错误的“父类”。这比裸露的 `except:` 要好，因为它不会捕获像 `SystemExit` 这样的系统级退出信号。



## 3. 黄金结构：`try...except...else...finally`



这是最完整的异常处理结构。

Python

```
try:
    # 1. [尝试] 存放主要风险代码
    print("1. [Try] 尝试...")
    num = int(input("输入非0数字: "))
    result = 10 / num

except ValueError:
    # 2. [如果失败] 处理 ValueError
    print("2. [Except] 输入的不是数字！")

except ZeroDivisionError:
    # 3. [如果失败] 处理 ZeroDivisionError
    print("3. [Except] 输入的是 0！")

else:
    # 4. [如果成功] 只有在 try 块 *没有* 异常时才执行
    #    (Plan A 成功后的收尾工作)
    print(f"4. [Else] 成功！结果是: {result}")

finally:
    # 5. [无论如何] 无论成功还是失败，*永远* 会执行
    #    (用于“清理”工作，如关闭文件)
    print("5. [Finally] 清理工作，无论如何都执行。")
```



## 4. 主动抛出：`raise`（“设置关卡”）



`raise` 关键字允许你**主动制造**一个错误。这对于在函数中验证输入（“卫兵语句”）非常有用。

- **目的：** “尽早失败” (Fail Fast)。与其让错误的数据在程序深处引发奇怪的 bug，不如在入口处就立刻用 `raise` 阻止它。

Python

```
def set_age(age):
    """设置年龄，但会验证。"""
    
    # 1. 验证类型
    if not isinstance(age, int):
        raise TypeError(f"年龄必须是整数 (int)，但收到的是 {type(age)}")
        
    # 2. 验证值
    if age < 0 or age > 150:
        raise ValueError(f"年龄必须在 0-150 之间，但收到的是 {age}")
    
    print(f"成功设置年龄为: {age}")

# --- 调用者来处理这些“关卡” ---
try:
    set_age("二十") # 触发 TypeError
    set_age(-5)    # 触发 ValueError
except (TypeError, ValueError) as e:
    print(f"[捕获成功] {e}")
```



## 5. 重新抛出：`raise`（“上报问题”）



有时，你想在 `except` 块中捕获一个错误，执行一些操作（比如记录日志），然后**再把这个错误扔出去**，让“上层”的调用者去处理。

- **目的：** “我处理不了，但我记录了，交给我的上级。”
- **语法：** 在 `except` 块中，单独使用 `raise` 关键字。

Python

```
def process_data():
    """内层函数：只记录，不处理。"""
    try:
        data = int("abc") # 制造一个 ValueError
    except ValueError as e:
        # 1. 在“本地”处理（例如记录日志）
        print(f"[日志] process_data 失败: {e}")
        
        # 2. “重新抛出”，把异常扔给调用者
        raise 
        
    # 'raise' 之后，这里不会执行
    print("process_data 成功")

# --- “上层”调用者 ---
try:
    process_data() # 调用内层
except ValueError:
    # 3. 成功捕获到了内层“重新抛出”的异常
    print("[上层] 已知晓并处理了 data_processing 的失败。")

print("--- 程序继续 ---")
```



### 重新抛出的关键点：



1. **流向：** 异常被抛向“上一层”调用者的 `try...except` 块。
2. **执行：** `raise` 会**立即停止** `except` 块的执行（但 `finally` 仍会执行）。
3. **冒泡：** 如果上一层也没有捕获，它会继续“冒泡”，直到顶层，导致程序崩溃。



## 6. 异常处理速查表



| **关键字** | **用法**                            | **何时执行？**                                               |
| ---------- | ----------------------------------- | ------------------------------------------------------------ |
| `try:`     | `try: <代码>`                       | **总是** - 首先尝试执行这里的代码。                          |
| `except:`  | `except <ErrorType> [as e]: <代码>` | **仅当** `try` 块中抛出了匹配的 `<ErrorType>` 异常时。       |
| `else:`    | `else: <代码>`                      | **仅当** `try` 块中**没有**抛出任何异常时。                  |
| `finally:` | `finally: <代码>`                   | **永远** - 无论 `try` 是成功、失败还是被 `return`，最后总会执行。 |
| `raise`    | `raise <ErrorType>("消息")`         | **立即** - 主动抛出一个新的异常。                            |
| `raise`    | `except: ... raise`                 | **立即** - 在 `except` 块中，重新抛出刚刚捕获的异常。        |

