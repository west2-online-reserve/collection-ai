# 编程学习笔记
## 1.基础容器：List（列表）、Dict（字典）的使用技巧
### list
* 基本操作
```
#长度
len(list) 

#删除元素
del list(n)

#列表加法
list1+list2

#map
list2 = map(lambda x: x * 2, list1)
list3 = map(lambda x, y: x + y, list1, list2)
```
* 列表推导式
```
# 列表推导式
list = [x**2 for x in range(10)]

# 带条件的列表推导式
list = [x**2 for x in range(10) if x % 2 == 0]
```
* 列表解包
```
# 基本解包
a, b, c = [1, 2, 3]

# 扩展解包
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```
* 列表切片
```
#切片操作符是在[]内提供一对可选数字，用:分割。冒号前的数表示切片的开始位置，冒号后的数字表示切片到哪里结束

# 反转列表
reversed_list = mlist[::-1]

# 获取最后n个元素
last_three = list[-3:]

# 每隔一个元素取一个
every_other = list[::2]

# 复制列表
list_copy = list[:]
```
* 列表排序和操作
```
# 排序
list = [3, 1, 4, 1, 5, 9, 2]
list.sort()   

# 自定义排序
words = ['apple', 'banana', 'cherry']
words.sort(key=len) # 按长度排序
words.sort(key=lambda x: x[1]) # 按第二个字符排序
```
* 列表元素计数
```
from collections import Counter
list = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
count = Counter(list)
# Counter({'apple': 3, 'banana': 2, 'orange': 1})
```
### dict
* 使用zip组合列表为字典
```
keys = ['a', 'b', 'c']
values = [1, 2, 3]
dict = dict(zip(keys, values))  # {'a': 1, 'b': 2, 'c': 3}
```
* 增加和删除
```
增加一项字典项的方法是，给一个新键赋值：
score['慕容复'] = 88
 删除一项字典项的方法是del：
del score['萧峰']
```
## 2.函数：Lambda 匿名函数、Decorator 装饰器
* lambda

lambda 参数列表: 表达式

它可以快速定义一个极度简单的单行函数，主要作用是内嵌到其他函数中
* decorator
装饰器是一种特殊的函数，用于动态地给其他函数添加功能，而无需修改原函数的代码
```
def 装饰器函数(被装饰函数):
    def 包装函数(*args,**kwargs):#*args,**kwargs可以传入任意个被包装函数的参数
        # 前置操作
        result = 被装饰函数(*args, **kwargs)  # 执行原函数
        # 后置操作
        return result # 必须在装饰器中返回函数的返回值，不然会报错
    return 包装函数

@装饰器函数
def 被装饰函数():
    # 原函数逻辑
    pass
```
## 3.面向对象：Class（类）与 Magic Methods（魔法方法），以及 OOP（面向对象编程）的思想
* OOP（面向对象编程）的思想

OOP 以 “现实世界中的事物” 为模型抽象出程序中的 “对象”
1. 封装：将数据（属性）和操作数据的方法（函数）捆绑在一个类中，隐藏内部实现细节，仅通过公开接口与外部交互。例如，一个 “人” 的类可以封装姓名、年龄（属性）和吃饭、睡觉（方法），外部无需知道 “吃饭” 的具体逻辑，只需调用方法即可。
2. 继承：子类可以继承父类的属性和方法，并在此基础上扩展或修改。减少代码重复，实现 “复用”。例如，“学生” 类可以继承 “人” 类的基本属性，再添加 “学号”“成绩” 等特有属性。
3. 多态：不同类的对象可以通过相同的接口（方法名）表现出不同的行为。例如，“猫” 和 “狗” 都继承自 “动物” 类，都有 “叫” 的方法，但调用时分别表现为 “喵喵” 和 “汪汪”。
* class
```
class Person:
    # 类属性：所有人类共享的特征
    species = "Homo sapiens"
    
    # 初始化方法：创建实例时初始化属性
    def __init__(self, name, age):
        self.name = name  # 实例属性：姓名
        self.age = age    # 实例属性：年龄
    
    # 实例方法：定义时,必须把self作为第一个参数，可以访问实例变量，只能通过实例名访问
    def greet(self):
        return f"Hello, I'm {self.name}, {self.age} years old."
    
    # 类方法：定义时,必须把类作为第一个参数，可以访问类变量，可以通过实例名或类名访问
    @classmethod
    def get_species(cls):
        return f"Species: {cls.species}"
    
    # 静态方法：独立于实例和类的工具函数.不强制传入self或者cls, 他对类和实例都一无所知。不能访问类变量，也不能访问实例变量；可以通过实例名或类名访问
    @staticmethod
    def is_adult(age):
        return age >= 18

# 创建对象（实例化）
person1 = Person("Alice", 30)
person2 = Person("Bob", 17)

# 访问实例属性和方法
print(person1.name)       # 输出：Alice
print(person1.greet())    # 输出：Hello, I'm Alice, 30 years old.

# 访问类属性和类方法
print(Person.species)     # 输出：Homo sapiens
print(Person.get_species())  # 输出：Species: Homo sapiens

# 调用静态方法
print(Person.is_adult(person2.age))  # 输出：False
```
* 魔法方法

魔法方法是 Python 中以双下划线 __ 开头和结尾的特殊方法，它们在特定场景下自动触发，用于自定义类的内置行为（如运算、打印、比较等）。
常用方法
1. 初始化与销毁	__init__, __del__	创建实例时初始化 / 销毁实例时清理
2. 字符串表示	__str__, __repr__	自定义输出
3. 运算符重载	__add__, __sub__	自定义 +/- 等运算符的行为
4. 属性访问控制	__getattr__, __setattr__	控制属性的获取 / 设置逻辑
5. 容器行为（列表 / 字典）	__len__, __getitem__	让类像列表一样支持 len()/[] 操作
## 4.文本处理：re 正则表达式
* 常用操作
1. 匹配（match 与 search）：
match 仅从字符串开头匹配，search 搜索整个字符串的首次匹配。
2. 查找所有匹配（findall）
3. 替换（sub）
* 元字符
1. .匹配任意单个字符（除换行符 \n）	a.b 匹配 aab、acb		
2. ^	匹配字符串开头	^hello 匹配以 hello 开头的字符串		
3. $	匹配字符串结尾	world$ 匹配以 world 结尾的字符串		
4. * 匹配前面的字符0 次或多次	ab* 匹配 a、ab、abb		
5. +匹配前面的字符1 次或多次	ab+ 匹配 ab、abb（不匹配 a）		
6. ?匹配前面的字符0 次或 1 次	ab? 匹配 a、ab		
7. {n}	匹配前面的字符恰好n 次	a{2} 匹配 aa		
8. {n,}	匹配前面的字符至少 n 次	a{2,} 匹配 aa、aaa		
9. {n,m}匹配前面的字符n 到 m 次	a{1,3} 匹配 a、aa、aaa		
10. []字符集，匹配其中任意一个字符	[abc] 匹配 a、b 或 c		
11. [^]	否定字符集，匹配不在其中的任意字符	[^abc] 匹配非 a、b、c 的字符		
12. ``逻辑 “或”，匹配左边或右边的模式	`a	b匹配a或b`
13. ()分组，将子模式视为整体	(ab)+ 匹配 ab、abab		
14. \转义字符，匹配元字符本身（如 \. 匹配 .）	a\.b 匹配 a.b
## 5.代码格式：列表推导式、Type Hint（类型注释）
* 列表推导式
```
[表达式 for 变量 in 可迭代对象 if 条件]

#表达式：生成新列表元素的计算逻辑）。
#变量：遍历可迭代对象时的临时变量。
#可迭代对象：被遍历的对象（如 list、range、str 等）。
#if 条件（可选）：筛选符合条件的元素，仅保留条件为 True 的元素。
```
* Type Hint（类型注释）: 标注变量、函数参数和返回值的类型。
```
（1）变量类型注释
# 格式：变量名: 类型 = 值
name: str = "Alice"       # 字符串
age: int = 25             # 整数
height: float = 1.65      # 浮点数
is_student: bool = True   # 布尔值
（2）函数参数与返回值注释
# 格式：def 函数名(参数: 类型) -> 返回值类型:
def add(a: int, b: int) -> int:
    return a + b

def get_greeting(name: str) -> str:
    return f"Hello, {name}!"
```
## 6.进阶技巧：generator 生成器（yield 关键字）
生成器是 Python 中一种特殊的迭代器，通过 yield 关键字实现，核心作用是惰性生成数据（按需产生值，不一次性占用大量内存）
```
def num_generator(n):
    for i in range(n):
        yield i  # 返回 i 并暂停，下次从这里继续

# 调用生成器函数，得到生成器对象
gen = num_generator(3)

# 迭代取值（每次取一个，用完即弃）
print(next(gen))  # 0（第一次执行到 yield）
print(next(gen))  # 1（从上次暂停处继续）
print(next(gen))  # 2
print(next(gen))  # 抛出 StopIteration（无更多值）
```