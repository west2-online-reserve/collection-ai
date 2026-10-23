# Python3 Decorator（装饰器）

> 内容来源：Task 0 函数计时练习，结合宝可梦 `@property` 用法整理。
> 语义核对：[Python 官方文档：函数定义与装饰器](https://docs.python.org/3.13/reference/compound_stmts.html#function-definitions)、[functools.wraps()](https://docs.python.org/3.13/library/functools.html#functools.wraps)。
> 整理日期：2026-10-21。

## 作用

装饰器接收一个函数，返回用于替换原名称的对象。常见用法是返回一个包装函数，在调用原函数前后增加日志、计时等公共行为。

求和逻辑可以继续只负责求和，计时逻辑留在装饰器中，减少重复代码。

## 语法

以下为语法示意，不作为完整程序运行：

```python
@decorator
def target(*args, **kwargs):
    ...

# 上面的装饰写法可理解为：
target = decorator(target)
```

这里的赋值用于解释含义，不能在已经装饰后再执行一次，否则会重复包装。

## 参数

- `decorator`：接收原函数的可调用对象。
- `func`：下方示例中用于接收原函数的参数。
- `*args`：把传入的位置参数收集为元组。
- `**kwargs`：把传入的关键字参数收集为字典。
- `func(*args, **kwargs)`：将收集到的参数展开，继续传给原函数。
- `wraps(func)`：创建帮助包装函数保留原函数元信息的装饰器。

## 返回值

需要区分两次返回：装饰器中的 `return wrapper` 返回包装函数；包装函数中的 `return result` 把原函数的结果交给调用方。

如果包装函数漏掉返回结果，调用方通常只会得到 `None`，即使原函数计算出了数值。

## 示例 1：在函数执行前后增加提示

```python
def show_call(func):
    def wrapper():
        print("开始")
        func()
        print("结束")
    return wrapper

@show_call
def greet():
    print("你好")

greet()
```

输出：

```text
开始
你好
结束
```

定义完成后，`greet` 指向包装函数。调用时先进入 `wrapper()`，它再通过闭包中保存的 `func` 调用原来的 `greet()`。

`return wrapper` 没有括号，因为这里需要交出函数本身，不是在定义阶段执行它。

## 示例 2：转交参数，并保留求和结果

```python
def show_call(func):
    def wrapper(*args, **kwargs):
        print("开始计算")
        result = func(*args, **kwargs)
        print("计算结束")
        return result
    return wrapper

@show_call
def add(a, b):
    return a + b

print(add(3, b=4))
```

输出：

```text
开始计算
计算结束
7
```

这里 `args` 为 `(3,)`，`kwargs` 为 `{'b': 4}`。参数被原样转交，所以包装后仍可混合使用位置参数和关键字参数。

计时练习可以在调用原函数前后分别读取 `time.perf_counter()`，再计算两次读数之差。具体耗时随运行环境变化，不适合写成固定预期数值。

## 示例 3：用 wraps 保留函数名称和说明

```python
from functools import wraps

def show_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@show_call
def add(a, b):
    """返回两数之和。"""
    return a + b

print(add.__name__)
print(add.__doc__)
print(add(2, 5))
```

输出：

```text
add
返回两数之和。
调用 add
7
```

不使用 `wraps()` 时，名称和说明通常会表现为包装函数自身的信息。`wraps()` 帮助保留原函数的元信息，但不会自动替我们完成参数转交或返回值处理。

本篇先展示包装机制，示例未展开通用装饰器的泛型类型声明；基础注解另见 [Type Hint](<Python3-TypeHint类型注释.md>)。

## 注意事项

- 装饰表达式在函数定义时处理，包装函数内部的代码通常在以后调用时执行。
- 多层装饰器相当于嵌套应用，例如两个装饰器可理解为 `outer(inner(func))`。
- 原函数抛出异常时，其后的普通语句不会执行；如果无论成功失败都要做清理，应使用 `try/finally`。
- `@property` 也是装饰器用法，但会形成属性访问接口，不能据此认为所有装饰器都把方法变成属性。
- 计算属性的项目例子见 [property 与计算属性](<Python3-property与计算属性.md>)。
