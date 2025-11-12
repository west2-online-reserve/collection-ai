## 基础容器



#### 列表

##### 定义语法

```python
# 语法：变量名 = [元素1, 元素2, 元素3]
list1 = [1, "good morning", [2, 3]]
list2 = [] # 定义空列表
```

##### 常用方法

```python
# 列表名[外层索引][内层索引]：修改指定索引处的元素
list1[2][0] = 22 

# .index(元素)：查找某元素在列表中的索引
list1.index(1) 

# .insert(索引，元素)：在指定索引处插入元素
list1.insert(1, 2)

# .append(元素)：将指定元素追加到列表尾部
list1.append([4, 5, 6])

# .extend(其它数据容器)：将其它数据容器的内容取出，依次追加到列表尾部

# del 列表[索引] / .pop(索引)：删除指定索引位置的元素

```



#### 字典

##### 定义语法

```python
# 变量名 = {key: value, key: value, ...}
dict1 = {1: "one", 2: "two", 3: "three"}
dict2 = {} # 定义空字典
```

字典不支持重复的key，如果有两个重复的key，会保留后一个

字典的key和value可以是任意数据类型(key不能为字典)



##### 用key获取value

```python
dict1 = {1: "one", 2: "two", 3: "three"}
print(dict1[1]) # 这里的1表示的是key
```



##### 嵌套字典

```python
student_score = {
    "A": {"Chinese": 140, "Math": 140, "English": 140},
    "B": {"Chinese": 142, "Math": 142, "English": 143},
    "C": {"Chinese": 145, "Math": 145, "English": 148},
}

print(f"A scored {student_score["A"]["Chinese"]} in Chinese")
```



##### 常用方法

```python
字典[key] = value：可以修改元素或者新增元素

字典.pop(key)：获得指定key的value，同时删除指定键值对

字典.clear()：清空字典

字典.keys()：获取字典中全部的key

len(字典)：计算字典中的元素数量
```



##### 字典遍历

```python
dict1 = {
    "A": {"salary": 3000, "level": 1},
    "B": {"salary": 4000, "level": 2},
    "C": {"salary": 5000, "level": 3},
    "D": {"salary": 3000, "level": 1},
    "E": {"salary": 6000, "level": 4},
}

# 遍历键和值
for key, value in dict1.items():
    print(f"员工{key}: 薪资{value["salary"]}, 级别{value["level"]}")
```

value：取到内层字典

value["salary"]：取到内层字典中salary对应的值



## 函数



#### lambda匿名函数

def关键字，可以定义有名字的函数

lambda关键字，可以定义没有名字的函数(匿名函数)

有名字的函数，可以根据名字重复使用

匿名函数，只可临时使用一次

```python
def test_func(calculate): # 形参是一个函数
    result = calculate(1, 2)
    print(result)

# 使用def关键字
def add(x, y):
    return x + y
test_func(add)

# 使用lambda关键字
# 语法：lambda 形参: 函数体(一行代码)
test_func(lambda x, y: x + y)
```



#### decorator装饰器

##### 不使用装饰器

```python
import time

def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    else:
        for i in range(2, num):
            if num % i == 0:
                return False
        return True

# 不使用装饰器        
def count_prime_nums():
    t1 = time.time()
    count = 0
    for i in range(2, 10000):
        if is_prime(i):
            count += 1
    t2 = time.time()
    print(f"total time: {t2 - t1}")
    return count 
# 这样写的话，这段代码既有计时的逻辑，又有检查质数的逻辑，就显得有点乱，所以就可以使用decorator

count = count_prime_nums()
print(count)
```



##### 使用装饰器

```python
import time

def display_time(func):
    def wrapper(*args): # 表示传入的函数有参数，但是不知道有几个
        t1 = time.time()
        total_number = func(*args)
        t2 = time.time()
        print(f"total time: {t2 - t1}")
        return total_number
    return wrapper
        
def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    else:
        for i in range(2, num):
            if num % i == 0:
                return False
        return True

# 使用装饰器
@display_time
def count_prime_nums(num):
    count = 0
    for i in range(2, num):
        if is_prime(i):
            count += 1
    return count

count = count_prime_nums(10000) # 执行的实际上是display_time这个函数
print(count)
```



## 面向对象



### 类与对象

#### 设计类

```python
class Student:
    # 类变量：属于类的变量，定义在方法外部，所有对象实例共享
    name = None
    
    def __init__(self, name):
        # 实例变量：属于特定对象实例的变量，每个对象实例都有自己独立的副本
        self.name = name
    
    # 实例方法
    def 方法名(self, 形参1, 形参2, ..., 形参N):
        方法体
        
    def say_hi(self):
        print(f"Hello, my name is {self.name}")
        
    def say_good_morning(self, str1):
        print(f"Hello, my name is {self.name}, {str1}")
```

##### 成员变量和成员方法

成员变量：类变量 + 实例变量

成员方法：类方法 + 实例方法 + 静态方法

##### self

self代表对象实例本身

通过self来访问实例变量和实例方法



#### 创建对象

```python
# 对象 = 类名()
student_1 = Student()

student_1.name = "A"

student_1.say_hi()

student_1.say_good_morning("good morning")
```



#### 类和对象

类是程序中的“设计图纸”

对象是根据图纸生产的具体实体

面向对象编程：设计类，创建对象，让对象做具体的工作



#### 构造方法

```python
# 构造方法：可以完成实例变量的声明和赋值
def __init__(self, name):
    self.name = name
```

**构造方法的特点**

在创建对象时，_ _ init _ _方法会自动执行

在创建对象时，将传入参数自动传递给_ _ init _ _方法使用



#### 魔术方法

Python类内置的方法，各自有各自特殊的功能，这些内置方法就称为“魔术方法”



##### str：实现类对象转字符串的行为

```python
class Student:
    def __init__(self, name, age):
        self.name = name
            
    def __str__(self):
        return f"name: {self.name}"
    
student_1 = Student("A")
print(student_1) # 输出：name: A
```



##### lt：用于两个类对象 < 和 > 的比较

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        
    def __lt__(self, other):
        return self.age < other.age

student_1 = Student("A", 18)
student_2 = Student("B", 19)
print(student_1 < student_2) # 输出：True
```



##### le：用于两个类对象 <= 和 >= 的比较

```python
def __le__(self, other):
    return self.age <= other.age
```



##### eq：用于两个类对象 = 的比较

```python
def __eq__(self, other):
    return self.age == other.age
```



### 面向对象编程的三大特性

#### 封装

![image-20251023120720843](C:\Users\17278\Desktop\python.assets\image-20251023120720843.png)



定义私有成员变量和私有成员方法时，只要在变量名和方法名前<u>加两个下划线</u>

私有成员变量和私有成员方法<u>无法被类对象直接使用</u>，但是<u>可以被其它成员使用</u>



#### 继承

继承表示的是，从父类那里复制成员变量和成员方法(不含私有)



##### 单继承

```python
class 类名(父类名):
    类内容体
```



##### 多继承

```python
class 类名(父类名1, 父类名2, ..., 父类名N):
    类内容体
    pass # 如果类内容体为空，就写个pass
```

如果有同名的变量和方法，先继承的优先级高于后继承的



##### 复写

子类继承父类的成员变量和成员方法后，如果对其不满意，可以进行复写

```python
class Phone:
    producer = "A"
    
    def call_by_5G(self):
        print("使用5G网络通话")
        
class MyPhone(Phone):
    producer = "B" # 复写成员变量
    
    def call_by_5G(self): # 复写成员方法
        print("开启CPU单核模式，确保通话时省电") 
        print("使用5G网络通话")
        print("关闭CPU单核性能，确保性能")
```



复写后，还是可以调用父类的变量和方法

```python
# 方式一
print(Phone.producer)
Phone.call_by_5G(self)

# 方式二
super().producer
super().call_by_5G()
```



#### 多态

多态指的是，同样的函数(同样的行为)，传入不同的对象，得到不同的状态



##### 多态作用在继承关系上

函数形参声明接收父类对象

实际传入子类对象进行工作

用以获得同一行为，不同状态

```python
class Animal:
    def speak(self):
        pass
        
class Dog(Animal):
    def speak(self):
        print("汪")
        
class Cat(Animal):
    def speak(self):
        print("喵")
        
def make_sound(animal: Animal):
    animal.speak()

dog = Dog()
cat = Cat()

make_sound(dog) # 输出：汪
make_sound(cat) # 输出：喵
```



##### 抽象类和多态配合使用

抽象类(接口)

```python
class Animal:
    def speak(self):
        pass
```

父类Animal的speak方法是空实现，这种设计的含义是：

父类用来确定有哪些方法

具体的方法实现，由子类自行决定



这种写法就叫做抽象类(也可以称之为接口)

抽象类就好比定义一个标准

包含了一些抽象方法，要求子类必须实现



抽象方法：方法体是空实现的方法，称之为抽象方法

抽象类：含有抽象方法的类，称之为抽象类

![image-20251025205603953](C:\Users\17278\Desktop\python.assets\image-20251025205603953.png)

抽象类的作用：

①多用于做顶层设计(设计标准)，以便子类做具体实现

②也是对子类的一种软性约束，要求子类必须复写(实现)父类的一些方法，并配合多态使用，获得同一行为的不同状态



## 文本处理



### re正则表达式

使用re模块，并基于re模块中三个基础方法来做正则匹配



#### 基础方法

##### re.match(匹配规则, 被匹配字符串)

从被匹配字符串开头进行匹配，匹配成功，则返回匹配对象(包含匹配信息)；匹配不成功，则返回空

是从被匹配的字符串的开头进行匹配，开头不匹配的话，后面匹配也不行

```python
import re

str1 = "good morning, good afternoon, good evening"

result1 = re.match("good morning", str1)
print(result1) # 输出：<re.Match object; span=(0, 12), match='good morning'>
print(result1.span()) # 输出：(0, 12)
print(result1.group()) # 输出：good morning

result2 = re.match("morning", str1)
print(result2) # 输出：None
```



##### re.search(匹配规则, 被匹配字符串)

从前往后匹配整个字符串，找到第一个匹配的后就停止；如果没有找到匹配的，就返回空

```python
import re

str1 = "good morning, good afternoon, good evening"

result1 = re.search("morning", str1)
print(result1) # 输出：<re.Match object; span=(5, 12), match='morning'>
```



##### re.findall(匹配规则, 被匹配字符串)

匹配整个字符串，找出所有匹配项；找不到匹配项，则返回空列表

```python
import re

str1 = "good morning, good afternoon, good evening"

result1 = re.findall("good", str1)
print(result1) # 输出：['good', 'good', 'good']
```



## 代码格式



#### 列表推导式

作用：用一种简明扼要的方式来创建列表

其实就是将常规写法压缩到了一行，然后将要append进列表的元素放在第一个位置



##### 简单模式

```python
# 常规写法
list_1 = []
for x in range(1, 10):
    list_1.append(x*x)
print(list_1) # 输出：[1, 4, 9, 16, 25, 36, 49, 64, 81]

# 列表推导式
list_1 = [x*x for x in range(1, 10)]
print(list_1) # 输出：[1, 4, 9, 16, 25, 36, 49, 64, 81]
```



##### 一般模式：包含判断和筛选

```python
# 常规写法
list_1 = []
for x in range(1, 10):
    if x % 2 == 0:        
        list_1.append(x*x)
print(list_1) # 输出：[4, 16, 36, 64]

# 列表推导式
#        step3      step1              step2
list_1 = [x*x for x in range(1, 10) if x % 2 == 0]
print(list_1) # 输出：[4, 16, 36, 64]
```



##### 变态模式：包含循环嵌套和判断筛选

```python
# 常规写法
list_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
list_2 = []
for y in list_1:
    for x in y:
        if x % 2 == 0:
            list_2.append(x)
print(list_2)# 输出：[2, 4, 6, 8]

# 列表推导式
list_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
list_2 = [x for y in list_1 for x in y if x % 2 == 0]
print(list_2)# 输出：[2, 4, 6, 8]
```



#### 类型注解

##### 作用

①帮助PyCharm等开发工具对代码做类型推断，协助做代码提示

②帮助开发者自身做类型的备注



##### 变量的类型注解

```python
# 基础数据类型的类型注解
name: str = "A"

#基础容器的类型注解
my_list: list = [1, 2, 3]

# 基础容器的类型详细注解
my_tuple: tuple[str, int] = ("good morning", 1)

# 类对象的类型注解
class Student:
    name = "A"
    
student_1: Student = Student()

# 在注释中进行类型注解
name = "A" # type: int

```

类型注解只是提示性的，并非决定性的

数据类型和注解类型不对应也不会报错



##### 函数/方法的类型注解

```python
# 形参的类型注解
def 函数名(形参名1: 类型, 形参名2: 类型):
    函数体
    
# 返回值的类型注解：
def 函数名(形参名1: 类型, 形参名2: 类型) -> 返回值类型:
    函数体
```



##### Union联合类型注解

```python
from typing import Union

my_list: list[Union[int, str]] = [1, 2, "Hello"]
```



## 进阶技巧



### generator生成器(yield关键字)

##### 如何让对象变成一个iterable对象？

方式一：添加getitem方法

```python
class MyList:
    def __init__(self, list1):
        self.list1 = list1

    def __getitem__(self, index):
        return self.list1[index]

my_list = MyList([1, 2, 3])
for x in my_list:
    print(x)
```



方式二：添加iter方法

```python
class MyList:
    def __init__(self, list1):
        self.list1 = list1

    def __iter__(self):
        return MyListIterator(self)

class MyListIterator:
    def __init__(self, my_list: MyList):
        self.my_list = my_list
        self.index = 0

    def __next__(self):
        if self.index >= len(self.my_list.list1):
            raise StopIteration
        result = self.my_list.list1[self.index]
        self.index += 1
        return result

my_list = MyList([1, 2, 3])

for x in my_list:
    print(x)
```



##### generator生成器(yield关键字)

```python
def my_range1(n): # my_range函数的返回值是一个可迭代对象list
    i = 0
    result = []
    while i < n:
        result.append(i)
        i += 1
    return result

for i in my_range1(2):
    print(i)

print(type(range(2))) # range不是函数，而是类，这里生成了range类的实例

# MyRange就是generator
class MyRange: # 作用：创建了MyRangeIter这个iterator
    def __init__(self, n):
        self.n = n

    # MyRange类的对象要放在for循环中，所以要在MyRange类中实现一个iter函数
    # iter方法需要返回一个iterator
    # 然后要实现MyRangeIter类，MyRangeIter就是iter方法返回的iterator
    def __iter__(self):
        return MyRangeIter(self.n)

class MyRangeIter: # 作用：把0,1,2,3,4含顺序返回
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __next__(self):
        if self.current >= self.n:
            raise StopIteration()
        result = self.current
        self.current += 1
        return result

for i in MyRange(2):
    print(i)

# 只要函数体中有yield关键字，返回值就会变成generator实例
# generator是由yield实现的一种特殊的iterable对象
def my_range2(n):
    print("enter my_range2")
    i = 0
    while i < n:
        yield i # 看见yield，函数就会暂停运行，next函数会让程序继续运行
        i += 1

result = my_range2(2) # 并没有进入my_range2函数，只是生成generator实例
it = result.__iter__() # 拿到my_range2的iterator
print(it.__next__()) # 到这里才进入my_range2函数
print(it.__next__())
print(it.__next__())
```

