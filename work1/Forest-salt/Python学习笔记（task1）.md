基础知识学习笔记
===
# 一、基础容器
## 1.List（列表）
```python
#创建列表
my_list=[1,2,3,'hello',True
empty_list=[]
list_from_constructor=list([1,2,3])
```
**特点：**
- 有序性：元素保持插入顺序
- 可变性：可以修改、添加、删除元素
- 允许重复：可以有重复元素
- 索引访问：支持正向和反向索引
### Python列表方法

| 方法 | 描述 |
|------|------|
|` append(x)` | 末尾添加元素 |
| `insert(i,x)` | 指定位置插入 |
| `remove(x)` | 删除第一个匹配元素 |
| `pop([i])` | 删除并返回指定位置元素 |
| `index(x)` | 返回元素索引 |
| `count(x)` | 统计元素出现次数 |
| `sort()` | 排序 |
| `reverse()` | 反转列表 |
| `copy()` | 浅拷贝 |

>**适用场景**
需要保持元素顺序的集合
需要频繁修改的序列
栈和队列的实现
需要索引访问的数据
## 2.Dict（字典）
```python
#创建字典
my_dict = {'name': 'Alice', 'age': 25, 'city': 'Beijing'}
empty_dict = {}
dict_from_constructor = dict(name='Bob', age=30)
```
**特点：**
- 键值对：存储键值对映射
- 可变性：可以添加、删除、修改键值对
- 键的唯一性：键必须唯一而且要可以哈希
- 值的任意性：值可以是任何类型

### Python字典方法
| 方法 | 描述 |
|------|------|
| `get(key[,default])` | 获取安全值 |
| `setdefault(key[,default])` | 获取值，不存在则设置 |
| `update([other])` | 批量更新 |
| `pop(key[,default])` | 删除并返回值 |
| `popitem()` | 删除并返回最后插入的键值对 |
| `keys()` | 返回所有键 |
| `values()` | 返回所有值 |
| `items()` | 返回所有键值对 |
| `clear()` | 清空字典 |

>**适用场景**
键值对映射关系
快速查找表
配置信息存储
JSON数据表示
# 二、函数
## 1.Lambda匿名函数
匿名函数是一种不需要使用def关键字定义函数名的简单函数。

**特点：**
- 没有函数名
- 使用`lambda`关键字创建
- 没有return，自动返回表达式的结果

**基本语法：**
```python
lambda 参数1, 参数2, ... : 表达式
```
举例：
```python
# 普通函数定义
def add(x, y):
    return x + y

# 等效的匿名函数
add_lambda = lambda x, y: x + y

# 使用
print(add(3, 5))        # 输出: 8
print(add_lambda(3, 5)) # 输出: 8
```
## 2.Decorator装饰器
Decorator是一个返回函数的高阶函数。
1. 举例
- 基础函数
```python
def say_hello():
    print("你好！")
    
say_hello()  # 输出：你好！
```
- 添加一个装饰器
```python
def my_decorator(func):  # 装饰器函数
    def wrapper():
        print("=== 函数开始执行 ===")  # 添加的功能
        func()  # 执行原始函数
        print("=== 函数执行结束 ===")  # 添加的功能
    return wrapper
```
- 使用装饰器
```python
@my_decorator  # 使用装饰器
def say_hello():
    print("你好！")
 
say_hello()
```
```python
#输出
=== 函数开始执行 ===
你好！
=== 函数执行结束 ===
```
2. 装饰器的语法糖
`@my_decorator`实际上等价于：
```python
def say_hello():
    print("你好！")

say_hello = my_decorator(say_hello)  # 手动包装
```
# 三、面向对象
## 1.Class类
1. 在Python中，定义类是通过class关键字：
```python
class Student(object):
    pass
```

2. 定义完了`Student`的类，就可以根据`Student`类创建出`Student`的实例，实例是通过类名+()实现的：
```python
bart=Student()
```
   变量`bart`指向的就是一个`Student`的实例

3. 可以自由地给一个实例变量绑定属性
```python
bart.name='Michael'
```

4. 类可以起到模板的作用，在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。通过定义一个特殊的`__init__`方法，在创建实例的时候，就把`name`，`score`等属性绑上去：
```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
```
`__init__`方法的第一个参数都是self，表示创建示例本身（可以类比于c++里的this指针）。因此在`__init__`方法内部，就可以把各种属性绑定到`self`，因为`self`就指向创建的实例本身。

有了`__init__`方法，在创建实例的时候，就不能传入空的参数了，必须传入与`__init__`方法匹配的参数，但`self`不需要传，Python解释器自己会把实例变量传进去：
```python
bart = Student('Bart Simpson', 59)
```

## 2.Magic Methods（魔法方法）
Magic Methods（魔法方法）是以双下划线开头、双下划线结尾的特殊方法。
### 常用的魔法方法
1.` __init__`-构造器
```python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

# 创建对象时自动调用
student = Student("Alice", 95)  # 自动调用 __init__
```
2. `__str__` - 字符串表示
```python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def __str__(self):
        return f"学生{self.name}，成绩{self.score}分"

student = Student("Alice", 95)
print(student)  # 输出：学生Alice，成绩95分
# 没有 __str__ 时输出：<__main__.Student object at 0x...>
```
## 3.OOP（面向对象编程）
面向对象编程----Object Oriented Programming，简称OOP，是一种程序设计思想。OPP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数。
在Python中，所有数据类型都可以被视为对象，也可以自定义对象。

**举例**
- 面向过程编程
```python
std1 = { 'name': 'Michael', 'score': 98 }
std2 = { 'name': 'Bob', 'score': 81 }
def print_score(std):
    print('%s: %s' % (std['name'], std['score']))
```
- 面向对象编程
要先将`Student`视为一个对象，这个对象拥有`name`和`score`这两个属性。如果要打印一个学生的成绩，首先必须创建出这个学生对应的对象，然后，给对象发一个`print_score`消息，让对象自己把自己的数据打印出来。
```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))
```

# 四、文本处理
## 1.re正则表达式
正则表达式是通过一种描述性的语言来给字符串定义一个规则，凡是符合规则的字符串，我们就认为它“匹配”，否则，该字符串就是不合法的。

### 正则表达式基础语法
1. 精确匹配
- **直接给出字符 = 精确匹配**
2. 特殊字符匹配
 - **正则表达式基础字符**
 
| 字符 | 描述 | 示例 |
|------|------|------|
| `\d` | 匹配一个数字 | `'00\d'` 匹配 `'007'`，不匹配 `'00A'` |
| `\w` | 匹配一个字母或数字 | `'\w\w\d'` 匹配 `'py3'` |
| `.` | 匹配任意字符 | `'py.'` 匹配 `'pyc'`、`'pyo'`、`'py!'` |
| `\s` | 匹配空格（包括Tab等空白符） | |
- **变长字符表示**

| 字符 | 描述 |
|------|------|
|`*`|表示任意个字符（包括0个）|
|`+`|至少要有一个字符|
|`?`|表示0个或1个字符|
|`{n}`|表示n个字符|
|`{n,m}`|表示n-m个字符|


3. 示例
`\d{3}\s+\d{3,8}`
```regex
\d{3}    # 匹配3个数字，如 '010'
\s+      # 匹配至少一个空格，如 ' ', '    '
\d{3,8}  # 匹配3-8个数字，如 '1234567'
```
# 五、代码技巧
## 1.列表推导式
列表推导式是一种创建列表的方法，它可以用一行代码完成原本需要多行循环才能实现的功能。
1. 基本语法
```python
[expression for item in iterable if condition]
```
- expression：对每个元素的操作
- item：循环变量
- iterable：可迭代对象（列表、range、字符串等）
- condition：可选的条件

2. 示例
- 传统循环
```python
 L = []
 for x in range(1, 11):
    L.append(x * x)
```
- 列表推导式
```python
[x * x for x in range(1, 11)]
```
for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方：
```python
[x * x for x in range(1, 11) if x % 2 == 0]
```
## 2.Type Hint（类型注释）
类型注释是为了让代码更清晰、更容易维护。
1. 基本语法
```python
变量名: 类型 = 值
```
2. 示例
- 变量类型注释：
```python
name: str = "Alice"
age: int = 25
height: float = 1.75
is_student: bool = True
```
- 函数类型解释
```python
def greet(name: str) -> str:
    return f"Hello, {name}"

def add_numbers(a: int, b: int) -> int:
    return a + b
```
- 复杂类型注释
```python
from typing import List, Dict, Tuple

# 列表类型
names: List[str] = ["Alice", "Bob", "Charlie"]
scores: List[int] = [95, 87, 92]

# 字典类型
student_info: Dict[str, str] = {"name": "Alice", "major": "Computer Science"}

# 元组类型（指定每个位置的类型）
coordinates: Tuple[float, float] = (40.7128, -74.0060)
```
- 可选类型注释
当某个值可能为None时，使用`optional：`
```python
from typing import Optional

def find_student(name: str) -> Optional[str]:
    """查找学生，可能找到也可能返回None"""
    # 模拟查找逻辑
    if name == "Alice":
        return "Alice found"
    else:
        return None
```

# 六、进阶技巧
## 1.generator生成器（yield关键字）
1. 生成器表达式
类似于列表推导式，但使用圆括号：
```python
# 列表推导式 - 立即创建整个列表
L = [x*x for x in range(1000000)]  # 占用大量内存

# 生成器表达式 - 按需生成
g = (x*x for x in range(1000000))   # 几乎不占内存

# 使用方式相同
for n in g:
    if n > 100:
        break
    print(n)
```
2. yield关键字的作用
这是定义`generator`的另一种方法。如果一个函数定义中包含`yield`关键字，那么这个函数就不再是一个普通函数，而是一个`generator`函数，调用一个`generator`函数将返回一个`generator`.
- 执行`yield`时，返回一个值
- 暂停执行函数，记住当前位置
- 下次请求时，从暂停的地方继续执行
```python
def simple_generator():
    print("开始执行")
    yield "第一个值"
    print("继续执行")
    yield "第二个值"
    print("结束执行")
    yield "第三个值"

gen = simple_generator()

print(next(gen))  # 输出：开始执行 → 第一个值
print(next(gen))  # 输出：继续执行 → 第二个值
print(next(gen))  # 输出：结束执行 → 第三个值
# print(next(gen))  # StopIteration 错误，没有更多值了
```



