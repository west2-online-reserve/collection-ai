---
tags: [python, 编程, 学习笔记]
date: 2024-01-01
aliases: [Python高级技巧]
---

# Python 进阶核心知识点笔记

## 目录
- [1. 基础容器高级用法](#1-基础容器高级用法)
- [2. 函数进阶特性](#2-函数进阶特性)
- [3. 面向对象编程](#3-面向对象编程)
- [4. 文本处理与正则表达式](#4-文本处理与正则表达式)
- [5. 代码格式与类型系统](#5-代码格式与类型系统)
- [6. 生成器与迭代器](#6-生成器与迭代器)

---

## 1. 基础容器高级用法

### 1.1 List（列表）高级技巧

```python
# 列表推导式 - 简洁高效创建列表
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# 嵌套列表推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 列表解包
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

# enumerate - 同时获取索引和值
fruits = ['apple', 'banana', 'cherry']
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")

# zip - 并行迭代多个列表
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
scores = [85, 92, 78]

for name, age, score in zip(names, ages, scores):
    print(f"{name}: {age}岁, 分数{score}")

# 列表切片技巧
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[::2])    # 每隔一个取一个: [0, 2, 4, 6, 8]
print(numbers[::-1])   # 反转列表: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

**技巧总结**：
- 列表推导式比传统 for 循环更简洁高效
- 解包操作可以灵活处理列表元素
- enumerate 和 zip 是迭代的利器
- 切片操作功能强大，支持步长和反向

### 1.2 Dict（字典）高级技巧

```python
# 字典推导式
square_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 安全的字典访问
person = {'name': 'Alice', 'age': 25}
age = person.get('age', 0)           # 25
city = person.get('city', 'Unknown')  # 'Unknown'

# setdefault设置默认值
person.setdefault('city', 'Beijing')
# 如果city不存在则设置为'Beijing'

# 字典合并 (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = dict1 | dict2  # {'a': 1, 'b': 3, 'c': 4}

# 字典解包传参
def print_person(name, age, city):
    print(f"{name}, {age}岁, 来自{city}")

person_info = {'name': 'Bob', 'age': 30, 'city': 'Shanghai'}
print_person(**person_info)
```

**要点**：
- 使用 `get()` 方法避免 KeyError 异常
- `setdefault()` 可以安全地设置默认值
- 字典推导式创建字典很高效
- 字典解包简化函数调用

---

## 2. 函数进阶特性

### 2.1 Lambda 匿名函数

```python
# 基本语法：lambda 参数: 表达式
add = lambda x, y: x + y
print(add(3, 5))  # 8

# 常用场景示例
numbers = [1, 4, 2, 8, 5, 3, 7, 6]

# 排序应用
sorted_numbers = sorted(numbers, key=lambda x: -x)
# 降序排列: [8, 7, 6, 5, 4, 3, 2, 1]

# 过滤应用
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
# [4, 2, 8, 6]

# 映射应用
squared = list(map(lambda x: x**2, numbers))
# [1, 16, 4, 64, 25, 9, 49, 36]

# 在字典排序中的应用
students = [
    {'name': 'Alice', 'score': 85, 'age': 20},
    {'name': 'Bob', 'score': 92, 'age': 22},
    {'name': 'Charlie', 'score': 85, 'age': 19}
]

# 多条件排序：先按分数降序，再按年龄升序
by_score_then_age = sorted(students, key=lambda x: (-x['score'], x['age']))
```

**应用场景**：
- 简单的函数逻辑，不需要定义完整函数
- 作为参数传递给高阶函数（map、filter、sorted）
- 代码简洁，一行完成简单操作

### 2.2 Decorator 装饰器

```python
# 基础装饰器
def simple_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"准备执行函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"函数执行完成: {func.__name__}")
        return result
    return wrapper

@simple_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# 输出:
# 准备执行函数: greet
# Hello, Alice!
# 函数执行完成: greet

# 保留原函数信息的装饰器
from functools import wraps

def logged(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"函数返回: {result}")
        return result
    return wrapper

@logged
def multiply(a, b):
    """乘法函数"""
    return a * b

print(multiply(3, 4))
# 调用函数: multiply
# 函数返回: 12
# 12

print(multiply.__name__)  # multiply
print(multiply.__doc__)   # 乘法函数
```

**装饰器用途**：
- 日志记录
- 性能测试
- 权限验证
- 输入验证
- 缓存结果

---

## 3. 面向对象编程

### 3.1 Class（类）基础与进阶

```python
from typing import Optional

class Person:
    # 类属性
    species = "人类"
    population = 0
    
    def __init__(self, name: str, age: int, email: Optional[str] = None):
        self.name = name
        self.age = age
        self.email = email
        Person.population += 1
    
    # 实例方法
    def introduce(self) -> str:
        return f"我叫{self.name}，今年{self.age}岁"
    
    # 类方法
    @classmethod
    def from_birth_year(cls, name: str, birth_year: int):
        from datetime import datetime
        age = datetime.now().year - birth_year
        return cls(name, age)
    
    # 静态方法
    @staticmethod
    def is_adult(age: int) -> bool:
        return age >= 18
    
    # 属性装饰器
    @property
    def is_adult_person(self) -> bool:
        return self.age >= 18
    
    # 字符串表示
    def __str__(self) -> str:
        return f"Person(name={self.name}, age={self.age})"
    
    def __repr__(self) -> str:
        return f"Person('{self.name}', {self.age})"

# 使用示例
person1 = Person("张三", 25)
print(person1.introduce())  # 我叫张三，今年25岁

person2 = Person.from_birth_year("李四", 1995)
print(person2.age)  # 根据出生年份计算的年龄

print(Person.is_adult(20))  # True
print(person1.is_adult_person)  # True
```

**方法类型总结**：
- **实例方法**：操作实例属性，第一个参数是 `self`
- **类方法**：操作类属性，第一个参数是 `cls`，用 `@classmethod` 装饰
- **静态方法**：与类相关但不操作类或实例，用 `@staticmethod` 装饰
- **属性方法**：像属性一样访问，用 `@property` 装饰

### 3.2 Magic Methods（魔法方法）

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    # 字符串表示
    def __str__(self) -> str:
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"
    
    # 算术运算
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)
    
    # 比较运算
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    def __len__(self) -> int:
        """返回向量的维度"""
        return 2
    
    def __getitem__(self, index: int) -> float:
        """支持索引访问"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")

# 使用示例
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)  # Vector(4, 6)
print(v1 * 2)   # Vector(2, 4)
print(v1 == Vector(1, 2))  # True
print(len(v1))  # 2
print(v1[0])    # 1
```

**常用魔法方法**：
- `__init__`: 构造函数
- `__str__`: 用户友好字符串
- `__repr__`: 开发者友好字符串
- `__add__`, `__sub__`, `__mul__`: 算术运算
- `__eq__`, `__lt__`: 比较运算
- `__len__`: 长度
- `__getitem__`: 索引访问

### 3.3 OOP 核心思想

```python
from abc import ABC, abstractmethod

# 抽象基类
class Animal(ABC):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    @abstractmethod
    def speak(self) -> str:
        pass
    
    @abstractmethod
    def move(self) -> str:
        pass
    
    def sleep(self) -> str:
        return f"{self.name}在睡觉"

# 继承
class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name}说: 汪汪！"
    
    def move(self) -> str:
        return f"{self.name}在跑"
    
    def fetch(self) -> str:
        return f"{self.name}在接飞盘"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name}说: 喵喵！"
    
    def move(self) -> str:
        return f"{self.name}在悄悄走路"
    
    def climb(self) -> str:
        return f"{self.name}在爬树"

# 多态演示
def animal_concert(animals: list[Animal]):
    for animal in animals:
        print(animal.speak())

def animal_activity(animals: list[Animal]):
    for animal in animals:
        print(animal.move())

# 使用示例
animals = [
    Dog("旺财", 3),
    Cat("咪咪", 2),
    Dog("小黑", 4)
]

print("=== 动物音乐会 ===")
animal_concert(animals)

print("\n=== 动物活动 ===")
animal_activity(animals)

# 类型检查和方法调用
for animal in animals:
    print(f"\n{animal.name}的活动:")
    print(animal.sleep())
    if isinstance(animal, Dog):
        print(animal.fetch())
    elif isinstance(animal, Cat):
        print(animal.climb())
```

**OOP 三大特性**：
- **封装**：将数据和方法包装在类中
- **继承**：子类继承父类的属性和方法
- **多态**：不同对象对同一方法有不同的实现

---

## 4. 文本处理与正则表达式

### 4.1 re 模块核心功能

```python
import re

text = """
联系人信息：
张三，电话：138-1234-5678，邮箱：zhangsan@example.com
李四，电话：139-8765-4321，邮箱：lisi@gmail.com
王五，电话：136-1111-2222，邮箱：wangwu@company.cn
"""

# 提取手机号码
def extract_phones(text: str) -> list[str]:
    phone_pattern = r'\b1[3-9]\d-\d{4}-\d{4}\b'
    return re.findall(phone_pattern, text)

# 提取邮箱地址
def extract_emails(text: str) -> list[str]:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

# 替换文本
def hide_phones(text: str) -> str:
    return re.sub(r'\b1[3-9]\d-\d{4}-\d{4}\b', '***-****-****', text)

# 分割文本
def split_by_punctuation(text: str) -> list[str]:
    return re.split(r'[，。！？]', text)

# 查找匹配位置
def find_phone_positions(text: str) -> list[tuple]:
    phone_pattern = r'\b1[3-9]\d-\d{4}-\d{4}\b'
    positions = []
    for match in re.finditer(phone_pattern, text):
        positions.append((match.group(), match.start(), match.end()))
    return positions

# 使用示例
print("提取的电话号码:", extract_phones(text))
print("提取的邮箱地址:", extract_emails(text))
print("隐藏电话号码后的文本:")
print(hide_phones(text))
print("按标点分割:", split_by_punctuation(text))
print("电话号码位置:", find_phone_positions(text))
```

**re 模块常用函数**：
- `re.findall()`: 查找所有匹配
- `re.search()`: 查找第一个匹配
- `re.match()`: 从字符串开头匹配
- `re.sub()`: 替换匹配文本
- `re.split()`: 按模式分割
- `re.finditer()`: 返回匹配迭代器

### 4.2 常用正则表达式模式

```python
import re

# 常用模式集合
patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b1[3-9]\d{9}\b',
    'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    'ip': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    'chinese': r'[\u4e00-\u9fff]+',
    'number': r'\b\d+\b',
    'date': r'\b\d{4}-\d{2}-\d{2}\b'
}

def extract_by_pattern(text: str, pattern_name: str) -> list[str]:
    """根据模式名称提取文本"""
    if pattern_name in patterns:
        return re.findall(patterns[pattern_name], text)
    return []

def validate_by_pattern(text: str, pattern_name: str) -> bool:
    """验证文本是否符合模式"""
    if pattern_name in patterns:
        return bool(re.fullmatch(patterns[pattern_name], text))
    return False

# 测试数据
test_text = """
用户信息：
姓名：张三，邮箱：zhangsan@test.com，电话：13800138000
网站：https://www.example.com，IP：192.168.1.1
日期：2024-01-15，金额：1000元
