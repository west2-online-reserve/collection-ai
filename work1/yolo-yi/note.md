# Python笔记

## 基础容器：List (列表)、Dict (字典) 的使用技巧

列表的形式和数组差不多，但列表里面可以放任意一种元素，还可以使用内置函数，如`append()`(直接添加元素)、`remove()`(删除元素)，`sort()`(可以直接对列表排序)等等。字典顾名思义就是通过一个索引来查找元素，字典里面包含的就是一个个键值对，键值对的第一个元素是索引，第二个是结果。

```python
# 比如以下的字典
dict={'name': 'Bob', 'age': 20}

# 使用键直接访问
print(dict['name']) #输出 Bob

# 使用 .get()方法
print(dict.get('age'))    #输出 20
print(dict.get('sex')) # 输出 None （键不存在返回None）
```

字典也有很多内置方法如`pop()`,`clear()`等

## 函数：Lambda 匿名函数、Decorator 装饰器

lambda函数是一行简单的，没有函数名的，临时使用的函数，在一些时候可以用来简化代码。

```python
# 形式为
lambda 参数列表: 表达式
```

以下是一些用法

```python
# 用作一个返回结果的简单函数
result = lambda x,y: x+y
print(result(2,5))   #输出 7

# 与内置函数一起使用
# 对列表的每个元素进行平方处理
num = [1, 2, 3, 4, 5]
square= list(map(lambda x: x**2, num))
print(square)   # 输出：[1, 4, 9, 16, 25]

# 按照年龄排序
stu = [("stu1", 20), ("stu2", 18), ("stu3", 24)]
sorted_stu = sorted(stu, key=lambda x: x[1])
print(sorted_stu)  # 输出：[('stu2', 18), ('stu1', 20), ('stu3', 22)]
```

装饰器本质上就是创建一个闭包函数，在闭包函数内调用目标函数，这样就能在不改变目标函数的同时，增加额外的功能

```python
def outer(func):
    def inner():
        print("我要睡觉了")
        func()
        print("我起来了")
    return inner

@outer
def sleep():
    import time
    print("睡眠中...")
    time .sleep(3)

sleep()
```

## 面向对象：Class (类) 与 Magic Methods (魔法方法)，以及 - OOP (面向对象编程) 的思想

把有相同属性和行为的对象放在一起就是一个类，比如人类，车类等，人类作为一个类就有相同的属性如性别，年龄等，以及相同行为如吃饭，睡觉等。

魔术方法是类内置的方法，以双下划线`__`开头，在定义对象和函数重载的时候会用到。比如`__init__()`（构造）,`__del__()`（析构）,`__str__()`（打印字符串）等。

面向对象编程思想的核心就是封装，继承，多态。
封装就是把对象的属性和行为封装在类中，这样就能隐藏里面的数据，保护隐私。继承就是一个子类直接继承父类的属性和行为，或者在父类的基础上再加上子类的特点，这样就能减少重复的代码。多态指一种行为在不同对象上有不同的表现，如动物发出声音这一个行为，在小猫上就表现为“喵喵喵”，在小狗上表现为“汪汪汪”等。

## 文本处理：re 正则表达式

re是python的一个模块，用`import re`引用，用来对文本进行处理。基础方法有`match`,`search`,`findall`等。

```python
# 以search为例
import re
s = "I like python"
result=re.search('python',s)
print(result.span())  # 输出 (7, 13)
```

然后就是元字符匹配，其中包含的字符很多，常用的示例就是作邮箱的验证

```python
import re
def Is_email(email):
    e = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(e,email))
emails = ['username@example.com', 'asdf.ds@dew', '123-abc.1@qe.co.uk']
for email in emails:
    print(f"{email}: {'有效' if Is_email(email) else '无效'}")
# 输出：
# username@example.com: 有效
# asdf.ds@dew: 无效
# 123-abc.1@qe.co.uk: 有效
```

## 代码美学：列表推导式、Type Hint (类型注释)

列表推导式是种一种让代码更美观和简洁的方法，可以用来创建列表

```python
# 传统的创建方式
list1 = []
for x in range(10):
    list.append(x**2)

# 列表推导式，代码更简洁
list2 = [x**2 for x in range(10)]
```

类型注释就是对变量、容器和函数返回值等作注释，如

```python
name :str ="Bob" # name的类型是字符串
age :int =18 # age的类型是整数
num :list[int] = [1, 2, 3] #num是列表

def greet(name: str) -> str: #函数返回str
    return f"Hello, {name}"

result: str = greet("Bob")
```

## 进阶技巧：generator 生成器 (yield 关键字)

生成器可以按需生成值，而不是一次性计算所有值，

```python
# 一次性返回所有结果
def square_list(list: list[int]):
    squared_list=[]
    for i in list:
        squared_list.append(i*i)
    return squared_list

list=[1, 2, 3, 4, 5]
squared_list=square_list(list)
print(squared_list)

#按需返回结果
def square_list(list: list[int]):
    for i in list:
        yield (i * i)

list=[1, 2, 3, 4, 5]
squared_list=square_list(list)
print(next(squared_list))
print(next(squared_list))
print(next(squared_list))
print(next(squared_list))
print(next(squared_list))
```
