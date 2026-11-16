# Python 语法与编程技巧学习笔记

# 简单的整理

## List

```python
numbers = [1, 2, 3, 4]
numbers.append(5)            # 在末尾添加5
numbers.insert(0, 0)         # 在index=0的地方插入0
numbers.pop()                # 删除最后一个元素
numbers[1:3]                 # 切片左闭右开
numbers[::-1]                # 反转
numbers.sort(key=lambda x, y: x + y) # 排序

first, *rest = [1, 2, 3]  # 解包 -> first=1, rest=[2,3]

for i, v in enumerate(["a", "b"]):  # 转化为索引 + 值
for a, b in zip([1,2], [3,4]):      # 并行遍历
```

## Dict
```python
person = {"name": "Alice", "age": 25}

age = person.get("age", 0)          # 获取

person["city"] = "Beijing"          # 添加或修改键值对
del person["age"]                   # 删除键值对

for key, value in person.items():   # 遍历字典
    pass

squares = {x: x**2 for x in range(5)} # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

## Lambda 匿名函数
```python
# 基础用法
add = lambda x, y: x + y
print(add(3, 5))  # 输出: 8

# 配合内置函数使用，比如 sorted 排序
data = [{"name": "A", "score": 90}, {"name": "B", "score": 85}]
# 按照 score 排序
sorted_data = sorted(data, key=lambda item: item["score"])
```

## Decorator 装饰器
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("--- 函数执行前 ---")
        result = func(*args, **kwargs)
        print("--- 函数执行后 ---")
        return result
    return wrapper

def say_hello(name: str):           # 先定义函数
    print(f"Hello, {name}!")

say_hello = my_decorator(say_hello) # 调用
say_hello("World")
```

## 面向对象: Class 与 Magic Methods
```python
class Dog:
    species = "cute dog"    # 类属性（所有实例共享）

    # 构造方法
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    # 魔法方法：定义 print()
    def __str__(self):
        return f"{self.name} is {self.age} years old."

    # 魔法方法：定义 len() 的内容
    def __len__(self):
        return self.age

my_dog = Dog("Buddy", 3)
print(my_dog)       # 输出: Buddy is 3 years old.
print(len(my_dog))  # 输出: 3
```

### re正则表达式
```python
import re

text = "我的电话是 123-456-7890，另一个是 987-654-3210"

#查找所有匹配的内容
phones = re.findall(r"\d{3}-\d{3}-\d{4}", text)
print(phones) # 输出: ['123-456-7890', '987-654-3210']

#查找第一个匹配项（返回 Match 对象）
match = re.search(r"\d{3}", text)
if match:
    print(match.group()) # 输出: 138

#替换匹配的内容
new_text = re.sub(r"-\d{4}", "", text) # 把 -xxxx 替换为空
```

## 列表推导式
```python
# [表达式 for 变量 in 对象]
squares = [x**2 for x in range(5)]              # [0, 1, 4, 9, 16]

# [表达式 for 变量 in 对象  可以带 if 条件]
evens = [x for x in range(10) if x % 2 == 0]    # [0, 2, 4, 6, 8]
```

## Type Hint 类型注释
```python
from typing import List, Dict, Optional

# 变量类型注释
age: int = 5
name: str = "keli"

# 函数参数和返回值注释
def greet(user_name: str, times: int = 1) -> str:
    return f"Hello {user_name} " * times

# 复杂类型注释
def process_data(data: List[int]) -> Dict[str, int]:
    return {"sum": sum(data), "count": len(data)}

# Optional 表示可以是该类型，也可以是 None
def get_user(user_id: int) -> Optional[str]:
    if user_id == 1:
        return "原来你是！！！"
    return None
```

## Generator 生成器
```python
def fibonacci_generator(n: int):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a  # 遇到 yield 会暂停函数，并返回 a 的值，下次调用从这里继续
        a, b = b, a + b
        count += 1

for num in fibonacci_generator(5):
    print(num) # 依次输出 0, 1, 1, 2, 3
```
