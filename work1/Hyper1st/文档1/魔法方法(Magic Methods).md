# 🐍 Python 魔法方法 (Magic Methods) 速查笔记



**“魔法方法”**（也称“Dunder 方法”，Dunder = **D**ouble **Under**score 双下划线）是 Python 中一类**自动调用**的特殊方法。它们以**两个下划线**开始和结束（如 `__init__`）。

它们存在的意义是**允许你自定义的类去“响应” Python 的内置语法和操作**。你从不（或极少）“手动”调用它们。

- 你**写**：`my_obj + other_obj`
- Python **自动调用**：`my_obj.__add__(other_obj)`
- 你**写**：`print(my_obj)`
- Python **自动调用**：`my_obj.__str__()`



## 1. 核心魔法方法

### a. 构造与初始化 (Initialization)



| **魔法方法**          | **自动调用时机**                                       | **目的 (你应在此方法中做什么)**                              |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| `__init__(self, ...)` | 当类的实例被**创建**时 (e.g., `my_obj = MyClass()`)    | **初始化**新创建的实例，设置其默认属性 (e.g., `self.name = ...`)。这是最常用的。 |
| `__new__(cls, ...)`   | 在 `__init__` *之前*被调用，是**真正创建**实例的方法。 | 控制实例的创建过程。**初学者极少需要重写此方法。**           |
| `__del__(self)`       | 当对象的引用计数为 0 (即将被垃圾回收) 时。             | “析构器”。用于执行清理工作。**不推荐依赖此方法。**           |



### b. 字符串表示 (String Representation)



**这是第二组最应掌握的方法。**

| **魔法方法**     | **自动调用时机**                       | **目的 (应返回什么)**                                        |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| `__str__(self)`  | `print(obj)`, `str(obj)`               | **“用户友好”** (User-friendly)。返回一个易读的、非正式的字符串。 |
| `__repr__(self)` | `obj` (在交互式 shell 中), `repr(obj)` | **“开发者友好”** (Developer-friendly)。返回一个**明确、无歧义**的字符串。 |

> **最佳实践：**
>
> 1. `__repr__` 的返回值最好是一个**可以被复制粘贴回 Python 并重新创建该对象**的有效代码（例如：`Car('Tesla', 'Model 3')`）。
> 2. **至少实现 `__repr__`**。如果 `__str__` 未定义，`print()` 会自动使用 `__repr__` 作为“备胎”。

Python

```
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        
    def __str__(self):
        return f"一辆漂亮的 {self.make} {self.model}"
        
    def __repr__(self):
        return f"Car('{self.make}', '{self.model}')"
        
my_car = Car("Tesla", "Model 3")
# print(my_car)  -> "一辆漂亮的 Tesla Model 3"  (调用 __str__)
# my_car         -> Car('Tesla', 'Model 3')    (在 shell 中, 调用 __repr__)
```



### c. 运算符重载 - 算术 (Arithmetic)



| **你的语法**   | **Python 调用的魔法方法** | **描述**     |
| -------------- | ------------------------- | ------------ |
| `obj1 + obj2`  | `obj1.__add__(obj2)`      | 加法 `+`     |
| `obj1 - obj2`  | `obj1.__sub__(obj2)`      | 减法 `-`     |
| `obj1 * obj2`  | `obj1.__mul__(obj2)`      | 乘法 `*`     |
| `obj1 / obj2`  | `obj1.__truediv__(obj2)`  | 真正除法 `/` |
| `obj1 // obj2` | `obj1.__floordiv__(obj2)` | 地板除 `//`  |
| `obj1 % obj2`  | `obj1.__mod__(obj2)`      | 取模 `%`     |
| `obj1 ** obj2` | `obj1.__pow__(obj2)`      | 幂 `**`      |



### d. 运算符重载 - 比较 (Comparison)



| **你的语法**   | **Python 调用的魔法方法** | **描述**                         |
| -------------- | ------------------------- | -------------------------------- |
| `obj1 == obj2` | `obj1.__eq__(obj2)`       | (Equal) 等于 `==`                |
| `obj1 != obj2` | `obj1.__ne__(obj2)`       | (Not Equal) 不等于 `!=`          |
| `obj1 < obj2`  | `obj1.__lt__(obj2)`       | (Less Than) 小于 `<`             |
| `obj1 > obj2`  | `obj1.__gt__(obj2)`       | (Greater Than) 大于 `>`          |
| `obj1 <= obj2` | `obj1.__le__(obj2)`       | (Less or Equal) 小于等于 `<=`    |
| `obj1 >= obj2` | `obj1.__ge__(obj2)`       | (Greater or Equal) 大于等于 `>=` |

> **注意：** `__eq__` 是最常被重写的。如果你定义了 `__eq__`，Python 也会提供一个默认的 `__ne__` (不等于)，但最好两个都定义以确保清晰。



### e. 模拟容器 (Emulating Containers)



这些方法让你的对象表现得像一个列表 (list)、字典 (dict) 或集合 (set)。

| **你的语法 / 内置函数**   | **Python 调用的魔法方法**           | **目的**                                    |
| ------------------------- | ----------------------------------- | ------------------------------------------- |
| `len(obj)`                | `obj.__len__(self)`                 | 让 `len()` 工作。**必须**返回一个非负整数。 |
| `obj[key]` (读取)         | `obj.__getitem__(self, key)`        | **(极其有用)** 让 `[]` 索引访问生效。       |
| `obj[key] = value` (写入) | `obj.__setitem__(self, key, value)` | 让 `[]` 赋值生效。                          |
| `del obj[key]`            | `obj.__delitem__(self, key)`        | 让 `del` 关键字生效。                       |
| `for item in obj:`        | `obj.__iter__(self)`                | 让 `for` 循环生效 (返回一个迭代器)。        |
| `if item in obj:`         | `obj.__contains__(self, item)`      | 让 `in` 关键字生效。                        |

Python

```
class MyList:
    def __init__(self, data):
        self._data = list(data)
    
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, index):
        return self._data[index]

my_list = MyList([1, 2, 3])
# print(len(my_list))  -> 3    (调用 __len__)
# print(my_list[0])    -> 1    (调用 __getitem__)
```



### f. 布尔值与调用 (Boolean & Call)



| **你的语法 / 内置函数**  | **Python 调用的魔法方法** | **目的**                                                     |
| ------------------------ | ------------------------- | ------------------------------------------------------------ |
| `if obj:` (布尔上下文)   | `obj.__bool__(self)`      | 定义对象何时被视为 `True` 或 `False`。**必须**返回 `True` 或 `False`。 |
| `obj()` (像函数一样调用) | `obj.__call__(self, ...)` | 允许你的**实例**被“调用”。                                   |

Python

```
class Counter:
    def __init__(self):
        self._count = 0
    
    # 像函数一样调用实例
    def __call__(self):
        self._count += 1
        print(f"被调用了 {self._count} 次")

c = Counter()
c() # -> "被调用了 1 次" (调用 __call__)
c() # -> "被调用了 2 次" (调用 __call__)
```

