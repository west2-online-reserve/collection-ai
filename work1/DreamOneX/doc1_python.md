# Python

## 基础容器：List & Dict 使用技巧

### List（列表）
- 创建与索引
```python
lst = [1, 2, 3]
print(lst[0])   # 1
```

- 列表合并与扩展
```python
lst += [4, 5]
lst.extend([6, 7])
```

- 常用方法
```python
lst.append(10)
lst.insert(1, 99)
lst.remove(2)
```

### Dict（字典）
- 创建与访问
```python
d = {"name": "Fall", "age": 20}
print(d["name"])         # Fall
```

- 遍历
```python
for k, v in d.items():
    print(k, v)
```

- 设置默认值（避免 KeyError）
```python
value = d.get("gender", "unknown")
```

## 函数：Lambda & Decorator

### Lambda 匿名函数
```python
add = lambda x, y: x + y
print(add(2, 3))   # 5
```

排序使用：
```python
nums = [(1, "b"), (3, "a"), (2, "c")]
nums.sort(key=lambda x: x[1])
```

### Decorator 装饰器
```python
def logger(func):
    def wrapper(*args, **kwargs):
        print("Calling:", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@logger
def hello():
    print("Hello")

hello()
```

## 面向对象：Class & Magic Methods

### Class（类）基础
```python
class Person:
    def __init__(self, name):
        self.name = name

    def say(self):
        print(f"I am {self.name}")

p = Person("Fall")
p.say()
```

### Magic Methods（魔法方法）
常见魔法方法：

| 方法 | 用途 |
|------|------|
| `__init__` | 构造函数 |
| `__str__` | 字符串显示 |
| `__len__` | 长度 |
| `__getitem__` | 使对象可索引 |
| `__iter__` | 可迭代 |

示例：
```python
class Bag:
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

bag = Bag([1, 2, 3])
print(len(bag))   # 3
print(bag[1])     # 2
```

## 面向对象思想（OOP）
- 封装：隐藏内部实现
- 继承：复用父类功能
- 多态：同名方法，不同行为

```python
class Animal:
    def sound(self):
        raise NotImplementedError

class Dog(Animal):
    def sound(self):
        return "wang"

class Cat(Animal):
    def sound(self):
        return "miao"
```

## 文本处理：re 正则表达式

```python
import re

text = "My phone is 123-4567"
m = re.search(r"\d{3}-\d{4}", text)
print(m.group())      # 123-4567
```

替换：
```python
re.sub(r"\s+", "_", "a   b c")
```

## 代码格式：列表推导式 & Type Hints

### 列表推导式
```python
squares = [x*x for x in range(5)]
```

带条件：
```python
even = [x for x in range(10) if x % 2 == 0]
```

### Type Hint（类型注释）
```python
def add(x: int, y: int) -> int:
    return x + y

from typing import List, Dict

nums: List[int] = [1, 2, 3]
info: Dict[str, int] = {"a": 1}
```

## 进阶技巧：Generator（生成器）与 yield

```python
def gen_numbers():
    for i in range(3):
        yield i

g = gen_numbers()
print(next(g))   # 0
print(list(g))   # [1, 2]
```

生成器表达式：
```python
g = (x*x for x in range(5))
```