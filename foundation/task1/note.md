# List
- ## **用法**
  - ### 创建
    - list=[]
    - 列表推导式 [表达式 for 变量 in 可迭代对象 if 条件]
  - ### 常用操作
    - #### 添加元素
      - numbers.append(4)          # 末尾添加
      - numbers.insert(0, 0)      # 指定位置插入
    #### 删除
      - numbers.pop()              # 末尾
      - numbers.remove(2)          # 删除第一个值为2的元素
    #### 切片
      - sub = numbers[1:3]         # 索引1到2
      - reverse = numbers[::-1]    # 反转
    #### 排序
      - numbers.sort()             # 原地升序
      - numbers.sort(reverse=True) # 降序
      - sorted_numbers = sorted(numbers)  # 返回新列表
---
# Dict
- ## **用法**
  - ### 创建
    - dict={}
    - a={'a':1,'b':2}
    - dict([('a',1), ('b',2)])
    - squares = {x: x**2 for x in range(5)}#字典推导式
  - ### 常用操作
    - #### 获取与修改
    - scores['Charlie'] = 90          # 新增或修改
    - print(scores.get('Alice'))      #
    - print(scores.get('David', 0))   # 带默认值
    #### 删除
    - del scores['Bob']
    - removed = scores.pop('Alice')   # 删除并返回值
    - last = scores.popitem()         # 删除并返回最后一个键值对
    #### 遍历
    - for key in scores:                    # 遍历键
    - for value in scores.values():         # 遍历值
    - for key, value in scores.items():     # 遍历键值对
    #### 合并
    - scores.update({'Eve': 88, 'Frank': 76})   # 更新/添加多个
---
# lambda 匿名函数
- ## lambda 参数:表达式
```python
def add(x, y):
    return x + y
add_lambda = lambda x, y: x + y
```
- ## 常用场景
```python
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
pairs.sort(key=lambda x: x[1])  # 按第二个元素排序
print(pairs)  # [(1, 'one'), (3, 'three'), (2, 'two')]
```
```python
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))  # [1, 4, 9, 16]
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4](常常和map，filter结合)
```
# Decorator
> 接受函数作为参数，但不改变原函数代码增加新功能
  - ## 结构
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # 调用前执行的操作
        result = func(*args, **kwargs)
        # 调用后执行的操作
        return result
    return wrapper
```
  - ## 范例
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时 {end-start:.3f} 秒")
        return result
    return wrapper

@timer 
def say_hello(name):
    time.sleep(1)
    print(f"Hello, {name}!")

say_hello("Alice")

@property#写pokemon.py用到的 ai提供的方法
    def effective_attack(self):
        """实际攻击力 = 基础攻击 + 火系叠层的加成

        火系被动：每造成一次伤害，攻击力 +10%，最多叠 4 层。
        比如基础攻击 35，叠了 2 层 → 实际攻击 = 35 + 35×0.1×2 = 42
        """
        return self.attack + int(self.attack * 0.1 * self.attack_stacks)
```
---
# OOP思想
- 主要分为**封装**，**继承**，**多态**
- ## 封装
  - 数据操作在对象内部，对外不显示出细节部分
- ## 继承
  - 子类对父类的属性，方法的继承，可以扩展新东西也可以重写
- ## 多态
  - 不同对象对于相同方法表现的行为不同
---
# 类

```python
class Dog:
    species = "Canis familiaris"  # 类属性
    def __init__(self, name, age):    # 构造方法（初始化）
        self.name = name              # 实例属性
        self.age = age
    def bark(self):                   # 实例方法
        print(f"{self.name} says Woof!")
my_dog = Dog("Rex", 3)
print(my_dog.name)
my_dog.bark()
```
>class Dog(Animal)继承
>super()调用父类
---

# 类的基础方法

## 实例方法、类方法、静态方法

```python
class Demo:
    def instance_method(self):
        # 最常用，self 指向实例
        return f"实例方法"

    @classmethod
    def class_method(cls):
        # cls 指向类本身，不依赖实例
        return f"类方法"

    @staticmethod
    def static_method():
        # 就是普通函数，只是放在类里归类
        return f"静态方法"

# 调用方式不同
d = Demo()
d.instance_method()        # 实例方法 → 要实例才能调
Demo.class_method()         # 类方法 → 类和实例都能调
Demo.static_method()        # 静态方法 → 类和实例都能调
```


## property（把方法当属性用）

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14 * self._radius ** 2

c = Circle(5)
print(c.area)    # 像属性一样调用，不用加括号
```

`@property` 把一个方法变成属性那样访问，调用时不用加 `()`。适合需要"算一下才出来"但又不想写成方法的场景。

---

# Magic Methods
- __内容__
- ## 常用方法
| 方法 | 用途 | 触发场景 |
|------|------|----------|
| `__init__(self, ...)` | 构造方法 | `obj = Class(...)` |（写类的时候最常用）
| `__str__(self)` | 用户友好字符串 | `print(obj)`、`str(obj)` |
| `__repr__(self)` | 开发者友好字符串（调试） | 交互式解释器显示 |
| `__eq__(self, other)` | 相等比较 | `obj == other` |
| `__lt__(self, other)` | 小于比较 | `obj < other` |
| `__len__(self)` | 长度 | `len(obj)` |
| `__getitem__(self, key)` | 下标访问 | `obj[key]` |
| `__setitem__(self, key, value)` | 下标赋值 | `obj[key] = value` |
| `__call__(self, ...)` | 对象可调用 | `obj(...)` |
| `__enter__` / `__exit__` | 上下文管理器 | `with obj:` |
---
# 正则表达式
- ## 常用re函数
| 函数             | 说明 |
|------------------|------|
| `re.match(pat, s)` | 从字符串开头匹配，返回匹配对象或 `None` |
| `re.search(pat, s)` | 搜索整个字符串，返回第一个匹配 |
| `re.findall(pat, s)` | 返回所有匹配的字符串列表 |
| `re.sub(pat, rep, s)` | 替换匹配的内容 |
| `re.compile(pat)` | 编译模式，提高重复使用效率 |
- ## 常用字符
| 符号 | 含义 | 示例 |
|------|------|------|
| `.` | 匹配任意字符（除换行） | `a.b` → `acb` |
| `\d` | 数字 (0-9) | `\d\d` → `12` |
| `\w` | 字母、数字、下划线 | `\w+` → `hello_123` |
| `\s` | 空白字符（空格、Tab、换行） | `a\sb` → `a b` |
| `^` | 字符串开头 | `^\d` → 以数字开头 |
| `$` | 字符串结尾 | `\d$` → 以数字结尾 |
| `*` | 重复 0 次或多次 | `a*` → 空、`a`、`aa`… |
| `+` | 重复 1 次或多次 | `a+` → `a`、`aa`… |
| `?` | 重复 0 次或 1 次 | `a?` → 空或 `a` |
| `{m,n}` | 重复 m 到 n 次 | `a{2,4}` → `aa`、`aaa`、`aaaa` |
| `[abc]` | 字符集，匹配括号内任一字符 | `[aeiou]` → 任意元音 |
| `[a-z]` | 范围 | `[0-9]` 等价 `\d` |
| `( )`   | 分组，可捕获匹配内容 | `(ab)+` |
| `\`     | 转义 | `\.` 匹配点号本身 |
- ## 作用中的举例
```python
import re

pattern = r'^[a-zA-Z0-9]{6,18}$'   # 长度6-18，只含字母和数字
def val_password(pwd):
    return bool(re.fullmatch(pattern, pwd))

print(val_password("abc123"))    
print(val_password("abc_123"))   
```
# 类的继承
| 概念 | 说明 |
|------|------|
| `class A(B)` | A 继承 B，拿到 B 的所有属性和方法 |
| `super().__init__()` | 调父类的构造方法，避免重复写 |
| 覆盖（override） | 子类重新定义父类已有的方法，比如 `begin()` |
| 类变量 | 写在类里面方法外面，所有实例共享，如 `type = "Fire"` |
| 实例变量 | `self.xxx`，每个实例自己一份，如 `self.evasion = 0.10` |



---

# 类型注解的循环引用（宝可梦踩的坑）

## 问题

```python
# pokemon.py
from effects import Effect     # 去 effects 拿 Effect

class Pokemon:
    def add_status(self, effect: Effect): ...  # 引用 Effect

# effects.py  
from pokemon import Pokemon   # 去 pokemon 拿 Pokemon

class Effect:
    def apply(self, pokemon: Pokemon): ...     # 引用 Pokemon
```

A import B，B import A，Python 死循环，报错。

## 两种解法

### 1️⃣ `from __future__ import annotations`

```python
from __future__ import annotations  # 加在文件最开头
from effects import Effect

class Pokemon:
    def add_status(self, effect: Effect) -> None:
        ...
```

作用：文件里所有类型注解都变成**字符串**，Python 不立即解析，等模块都加载完了再慢慢算。适合**自己引用自己**的文件（pokemon.py 里 `opponent: Pokemon`）。

### 2️⃣ TYPE_CHECKING

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pokemon import Pokemon   # 只在类型检查时执行

class Skill:
    def execute(self, user: "Pokemon") -> None:
        #                 ^^^^^^^^ 必须加引号！
        ...
```

`TYPE_CHECKING` 运行时是 `False`，里面的 import 不执行。但类型注解 `user: "Pokemon"` 是字符串，不会立即被 Python 解析。

三种写法：

- `from __future__ import annotations` → 文件里自己引用自己（pokemon.py）
- `TYPE_CHECKING` + `"引号"` → 引用外部类，运行时不需要（skills.py）
- 直接 import 不加保护 → 运行时确实需要那个类（play.py import Pokemon）

---

# Type Hint
- 提高代码可读性，方便检查，python本身没有强制类型
- ## 基本语法
  - def 函数名(...) -> 返回类型:
- ## 举例(from deepseek)
```python
name: str = "Alice"
age: int = 30
height: float = 1.75
is_student: bool = False
def greet(person: str, times: int = 1) -> str:
    return f"Hello, {person}! " * times
# 复杂类型
from typing import List, Dict, Tuple, Optional, Union
scores: List[int] = [95, 87, 76]
person: Dict[str, int] = {'Alice': 95}
pair: Tuple[str, int] = ("Alice", 95)
def find(key: str, data: Dict[str, int]) -> Optional[int]:
    return data.get(key)          
def handle(value: Union[int, str]) -> None:
    print(f"Got {value}")
def process(items: List[int]) -> List[str]:
    return [str(x) for x in items]
```
# generator and yield
- ## 含义:
  - 作为一个惰性求值迭代器，在每次需要的时候才计算产生值，节省内存。
  - ## 举例
```python
def simple_gen():
    print("开始")
    yield 1
    print("继续")
    yield 2
    print("结束")

gen = simple_gen()
print(next(gen))   # 打印 "开始" 然后返回 1
print(next(gen))   # 打印 "继续" 然后返回 2
print(next(gen))   # 打印 "结束" 并抛出 StopIteration
```
- ## 对比
  - 与正常的生成器对比，正常的生成器一次性会生成所有元素，占用内存，而yield关键字的逐个生成，不保存历史,几乎不占内存
---

# random 随机库

- ## 导入
  - `import random`

- ## 常用函数

  ```python
  import random

  # 随机小数
  print(random.random())          # [0.0, 1.0) 之间的随机浮点数
  print(random.uniform(1, 10))    # [1, 10] 之间的随机浮点数

  # 随机整数
  print(random.randint(1, 6))     # [1, 6] 之间的随机整数（闭区间）
  print(random.randrange(0, 10, 2))  # [0, 10) 步长为2的随机整数 → 0,2,4,6,8

  # 序列随机
  fruits = ['apple', 'banana', 'cherry', 'date']
  print(random.choice(fruits))        # 随机选一个元素
  print(random.choices(fruits, k=2))  # 随机选 k 个元素（可重复）
  print(random.sample(fruits, k=2))   # 随机选 k 个元素（不重复）

  # 打乱顺序
  cards = ['A', '2', '3', '4', '5']
  random.shuffle(cards)               # 原地打乱
  print(cards)

  # 设置种子（结果可复现）
  random.seed(42)
  print(random.randint(1, 100))       # 只要种子一样，结果就一样
  ```
