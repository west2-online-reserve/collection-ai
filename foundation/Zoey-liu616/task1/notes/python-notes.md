# 基础容器

## List列表

1. 空列表：`my_list = []` 或 `my_list = list()`

1. 列表下标索引可取出特定位置数据

    * 正数：从0开始递增(从左到右) 负数： 从-1开始递减（从右到左）

```python
# 创建一个嵌套列表；外层是list列表，内层是Tuple元组。
list1 = [(1, 2), (5, 9), (7, 0)]

# 先取第三个元素，再取第二个元素（第三个括号为[2],括号里面的[1]即定位到0）
print(list1[2][1])

# 输出结果为0
```

1. 功能

* 查询下标索引：语法：

    list1.index(元素)
* 修改：语法：

     list[下标] = 新值
* 插入元素： 语法：

    list1.insert(即将要成为的下标， 指定元素)
* 追加： 语法：
 list1.append(元素)

 append()在列表末尾添加一个元素

or  list.extend(其他数据容器/其他可迭代对象)

extend()将另一个可迭代对象中的元素依次添加到列表末尾

将其他数据容器内容取出，依次追加到列表尾部

* 删除：

    删除元素语法：
del list1[下标] or list1.pop(下标)

    删除某元素在列表中的第一个匹配项：
list1.remove(元素)

    清空：
list1.clear()

* 统计：

    统计某元素在列表中的数量：语法：list1.count(元素)

    统计共有几个元素：语法：len(list1)

## Dict字典

1. 含义：

通过key（键）去检索value（值）（类似于字典中通过字去检索含义）

* 每一个Key对应一个Value。

* Key不能重复。

* 字典中的数据以key:value的形式存储

空字典：

my_dict = {}

 my_dict = dict()

```python

student = {

    "name":"Amy",

    "age":18,

    "score":90
}#创建字典

student["name"]#通过name获取Amy

student.get("age")#获取18

```

* 注意：字典里面key和value要一一对应，括号里面的形式为：{key1 : value1, key2 : value2}  （key1:value1被称作一个元素）

1. 数据获取：

字典["key"]
嵌套字典也用 `{}` 定义，访问时可以连续写 `字典["key1"]["key2"]`

1. 常用操作：

* 新增/更新元素：
字典["key(新的)"] = "Value"

如果key不存在，则新增元素；如果已经存在，则修改对应的value

* 删除元素：
dict1.pop(key)

* 清空元素：
dict1.clear()

* 获取全部key:
dict1.keys()

* 遍历：

```python
for key, value in 字典.items():#items可以同时获取字典中的key和value

print(key, value)

```

先拿key，再通过字典[key]找对应value

```python

#举例
student = {"name": "Amy","age":18, "score": 90}

print(student["name"])#获取数据

student["class"] = "Python"#新增数据

student["age"] = 19#修改数据

for key, value in student.items():

    print(key, value)#遍历字典

```

## Lambda匿名函数

1. 定义：函数作为参数传递，传入计算逻辑（Lambda没有普通函数名称，用于简单计算逻辑，结果会自动返回，可较为简洁创建函数）

* 语法： lambda 参数：表达式（即为一行代码）
        test(lambda x, y: x+y)

```python
double = lambda x: x * 2

print(double(5))  # 10

```

Lambda 只是没有函数名的简短函数，并不是“只能使用一次”；像上面的 `double` 一样保存到变量后也能重复调用。

* 结合map()使用

```python

nums = [1, 2, 3, 4]

squared_nums = list(map(lambda x: x **2, nums))

print(squared_nums) #输出[1, 4, 9, 16]

```

* 结合filter()使用

```python

nums = [1, 2, 3, 4]

even_nums = list(filter(lambda x: x % 2 == 0, nums))

print(even_nums)#输出[2,4]

```

## Decorator装饰器

1. 定义： 不改原函数代码，给函数额外加功能

1. 基本结构

```python

def decorator(func):#func为传入的原函数

    def wrapper():#wrapper用来包装函数，增加额外功能

        print("开始执行")

        func()#执行原来的函数

        print("执行结束")

    return wrapper#返回包装后的函数

@decorator#使用装饰器；定义完成后等价于 hello = decorator(hello)
def hello():

    print("Hello Python!")

hello()#输出：开始执行 Hello Python! 执行结束

```

## 面向对象

1. Class类：把数据（name age等变量）与功能（def introduce()：这种函数）装在一起

* 解释：学生都有abc三个共同特质，故定义一个“学生类”（这个学生类就是一个模板）

```python
class Student:

    pass

student1 = Student() #创建真学生（即对象）

```

补充：__init__ 创建对象时自动执行的初始化函数

```python

class Student:

    def __init__(self, name, age):#name和age都是形参

        self.name = name

        self.age = age #self指的是当前对象自己

```

类是模板，按照类创建出来的 `student1` 是对象（实例）。

* 继承：继承父类的某些特征，同时可以在父类的基础上修改

单继承：`class Dog(Animal):`  # Dog 继承 Animal

多继承：继承多个父类 class 类名(父1，父2，父3)：

调用父类成员：（1）成员变量：父类名.成员变量

（2）成员方法：父类名.成员方法(self)

super().成员变量

super().成员方法()

```python
class A:#父类

    def test(self):

        print("A")

class B(A):#继承父类A

    def test(self):#调用

        print("B")

b = B()

b.test()

```

* 封装：把数据和操作这些数据的方法放到同一个类中

```python

class Pokemon:

    def __init__(self, hp):

        self.hp = hp

    def receive_damage(self, damage):

        self.hp -= damage#hp和扣血方法都在Pokemon类中，此为封装

```

* 多态：不同对象可以使用相同的方法名称，但表现出不同的行为

e.g:火属性宝可梦有自己的type_effectiveness(),草属性宝可梦有自己的type_effectiveness(),虽然它们调用同一个方法，但是返回的值不同

* 抽象：隐藏具体实现细节，让user关注需要完成什么功能

```python

def initialize_skills(self):

    raise NotImplementedError("子类需要实现技能初始化")#父类规定子类需要实现技能初始化，具体技能由子类自己决定

```

1. Magic Methods 魔法方法：

* __init__() 初始化对象时自动执行的函数，用来设置对象的属性

```python

class Student:

    def __init__(self, name, age):#name和age都是形参

        self.name = name

        self.age = age #self指的是当前对象自己

```

* __str__()控制对象转换成字符串时显示的内容，通常用于print()，且必须返回字符串

```python

class Student:

    def __init__(self, name):

        self.name = name

    def __str__(self) -> str:

        return self.name

student = Student("Amy")

print(student)

```

* __repr__()提供对象的详细表示，方便调试

* __lt__()小于

```python

def __lt__(self, other):#other传参

    return self.age < other.age

```

* __le__()小于等于

* __eq__()等于

* __len__()自定义len()的行为

## 文本处理：re正则表达式

1. re后面的类似于按规则找，使用之前先写import re

| 符号/常用函数 | 作用 |
|---|---|
| re.search() | 整段中从前往后找第一个符合条件的内容 |
| re.findall() | 找所有符合条件的 |
| re.match() | 只从字符串开头找（返回的不是字符串本身，而是匹配对象）|
| .group() | 用这个把真正匹配的对象取出来|
| \d | 匹配一个数字 |
| \w | 匹配字母、数字或下划线等字符 |
| \w+ | 可以匹配 hello |
| \s | 匹配空白字符 |
| + | 前面内容出现 1 次或多次 |
| * | 匹配零个或多个，前面的内容出现0次或多次 |
| . | 匹配除换行符外的任意一个字符 |
| ? | 前面的内容出现零次或一次 |
| ^ | 匹配字符串开头 |
| $ | 匹配字符串结尾 |

## 代码格式

1. 列表推导式(用一行代码快速创建一个列表)

```python

#未使用列表推导式
numbers = []

for i in range(1,6):

    numbers.append(i * i)

print(numbers)#输出[1, 4, 9, 16, 25]

```

```python

numbers = [i * i for i in range(1, 6)]#使用列表表达式

print(numbers)#输出[1, 4, 9, 16, 25]

```

* 有条件 [表达式 for 变量 in 可迭代对象 if 条件]

* 无条件 [表达式 for 变量 in 可迭代对象]

```python

num = [ i  for i  in range(1,11) if i % 2 == 0]

print(num)#输出[2, 4, 6, 8, 10]

```

1. Type Hint(类型注释，即告诉别人一个变量、函数参数或者返回值应该是什么类型)

Type Hint 只提供类型提示，方便阅读和静态检查；Python 运行时不会自动强制变量必须是该类型。

* 形式： 参数：类型

函数 -> 类型

```python

name: str = "Pikachu"#示例：变量类型注释

alive: bool = True

```

* int 整数

* float 小数

* str 字符串

* bool 布尔类型（True False）

* list[str] 字符串列表

* dict[str, int] 键为字符串，值为整数的字典

```python
def func(a: int, b: int) -> int:#参数a应该是整数，参数b也是整数，预计返回一个整数

    return a + b#函数类型注释

def show(name: str) -> None:#None类型注释，None表示函数没有返回有意义的结果

    print(name)

```

```python

names: list[str] = ["Amy", "Tom"]#list[str]是字符串列表

scores: dict[str, int] = {#键为字符串，值为整数的字典
    "Amy": 90,
    "Tom": 85
}
```



## 进阶技巧

1. generator生成器（yield 关键字）

* 定义：可以逐个产生数据的对象，不需要一次性把所有数据储存起来

* return 返回一次性结果

* yield 返回一个值，同时暂停函数执行，下次执行时从上次暂停的位置继续运行

```python

def generate_numbers():
    yield 1
    yield 2
    yield 3

for number in generate_numbers():
    print(number)#输出1 2 3
```

`return` 会结束函数并一次返回结果；`yield` 每次产生一个值后暂停，下一次迭代时从暂停的位置继续。
