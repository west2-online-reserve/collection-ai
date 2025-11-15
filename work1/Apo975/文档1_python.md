# Python基础知识点

## 基础容器

### List

List是一种有序集合，可以同时存储不同类型的元素，甚至可以存储另一个List

```pyhton
Lst=[1,'a',5.4321,'曼波',[1,2,3]]
```

使用len()方法可以获得List的长度（即元素个数，可为0）

```python
Len_Lst=len(Lst)
```

可以通过索引访问与操作List的元素

索引从0开始，若为负数如-n，则表示倒数第n个元素

```python
Lst[0]=3 #将List第一个元素1修改为3
Lst[-1]=[1] #将List倒数第一个元素[]修改为[1]
```

append()方法：在List末尾添加元素

insert()方法：在List指定位置插入元素

```python
Lst.append('m') #在List最后增加元素'm'
Lst.inser(1,'b') #在List原先索引为1的位置前插入元素'b'
```

pop()方法：删除List指定位置的元素（默认为最后一位）

```python
Lst.pop() #删除List最后一位元素
Lst.pop(2) #删除List原先索引为2的元素
```

切片：在List的 [ ] 里用 : 分割不同数字，可以对List进行切片

```python
Lst2 = Lst[i:j] #获得索引为[i,j)范围内的切片,注意左闭右开，不包含j
	Lst[0:3] #获得索引从0到2的切片
    Lst[:3] #获得索引从0到2的切片(简易写法)
    Lst[2:-2] #获得从索引为2的元素到倒数第3个元素的切片
    Lst[:101:2]#索引从0到100，逐次间隔1个元素进行切片
```

推导式：对列表元素进行描述，以获得最终列表

```python
L1=list(range(2,20,3))
L2=[x*2 for x in L1 if x > 5]
# L2 = [16, 22, 28, 34]
```



### Dict

dict即dictionary，使用key-value进行元素的储存，方便查找

```python
dic={'key':'value','a':10,1:2} 
#按照key:value格式存储元素，以','进行分隔
#key和value的数据类型不定
print(dic['key']) #输出key对应的value值
```

使用get方法可以更安全的访问dict元素：

```python
dic.get('key')
```

使用key可以直接修改对应的value值

```python
dic['a']=20
#此时dic的内容为：{'key':'value', 'a':20, 1:2}
```

直接指定新key对应的value，即可在字典中加入数据

```python
dic[114]=514
#此时dic的内容为：{'key':'value', 'a':20, 1:2, 114:514}
```

使用pop(key)或del可以删除dict中的元素：

```python
dic.pop('a') #或 del dic['a']
#此时dic的内容为：{'key': 'value', 1: 2, 114: 514}
```



## 函数

### Lambda

用于创建小型、一次性的匿名函数。

```python
sum = lambda a,b,c:a+b+c
# lambda 元素 : 返回值
sum(1,2,3) #返回1+2+3的结果
```



```python

```



### Decorator

Decorator是一种可以返回函数的高阶函数，可以在不改变函数内容的情况下，对函数的功能进行调整，起到对函数的”装饰“作用

```python
def decorator_name(func): #获取func
    
    def wrapper(*args, **kwargs): #包装函数
       	#
    	# 增加的功能（func运行前）
    	#
    	result = func(*args, **kwargs)
        #
    	# 增加的功能（func运行后）
    	#
        return result
    
    return wrapper  # ③ 装饰器返回包装函数
```

利用@将该装饰器应用于目标函数：

```python
@decorator_name
def func():
    print('Hello World')
    
# 实质是“新函数 = 装饰器(原函数)”
```



## 面向对象

### Class

使用class将数据和方法抽象整合封装起来，定义了一种抽象模板，即为类

```python
class Myclass:
    name = 'Bob'  # 类变量

    def fun(self):  # 类方法
        print(f"Hello,{self.name}")
```

通过类可以创建对应的对象，对象可以使用此类中的类变量与类方法

```python
ea = Myclass()  # 创建类对象
print(ea.name) #通过对象使用类变量
ea.fun() #通过对象使用类方法
```



### Magic Methods

Magic Methods 是以双下划线为开头和结尾的方法，它不供使用者直接调用，而由python解释器自动调用

```python
class Vector:
    def __init__(self, x, y): #__init__：初始化Vector对象
        self.x = x
        self.y = y
        
    def __str__(self): #__str__：用于print输出的友好格式
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other): #__add__：实现 '+' 运算符重载
        return Vector(self.x + other.x, self.y + other.y)
```

对此类的每一个对象进行操作时，都会调用对应的Magic Methods

```python
# 创建对象 (自动调用 __init__)
v1 = Vector(2, 3)
v2 = Vector(5, 7)

# 输出对象 (自动调用 __str__)
print(v1) #输出：Vector(2, 3)

# 使用 '+' (自动调用 __add__)
v3 = v1 + v2
print(v3) #输出：Vector(7, 10)
```



### OOP

面向对象思想的核心在于将现实世界中的事物抽象成程序中的对象，从而使代码结构更贴近真实世界，提高软件的可维护性、灵活性和可重用性

它主要通过下面几个内容实现：封装，继承，多态

封装的主要思想是：将数据和操作数据的方法绑定在一起，形成一个密不可分的对象。同时，隐藏对象的内部工作细节，只暴露必要的接口。

继承的主要思想是：允许新的类（子类）重用现有类（父类）的属性和方法。

多态的主要思想是：允许不同的对象对同一个方法调用作出不同的响应。

#### 封装

python对封装的规定并不严格，仅仅使用下划线作为约定，适当限制内部属性的使用

在定义类时，在变量或者方法的名字前加入下划线"_"或双下划线"__"，即可对其进行封装，不易被类外直接使用

```python
class Myclass:
    __name = 'Bob'

    def fun(self):
        print(f"Hello,{self.__name}") #在类方法中可以调用被保护的类变量__name

ea = Myclass()
ea.fun() #通过类方法调用__name
print(ea.__name) #运行报错，因为类外无法直接调用__name
```

#### 继承

定义类时，可以在括号中指定父类，使新的类继承父类的属性和方法，同时可以定义独属于子类的属性与方法

```python
class Baseclass:
    name = 'Bob'
    def fun(self):
        print(f"My name is,{self.name}")

class Subclass(Baseclass): #子类继承父类
    age = 18
    def fun2(self):
        print(f"My age is {self.age}")

eb=Subclass()
eb.fun() #子类对象可以调用父类方法
eb.fun2()
```

#### 多态

在python中想要实现多态（用同一个函数处理不同对象）只需要利用“Duck Typing"机制：

如果多个对象有同名的方法，那我们可以对它们直接调用这个方法，而不需要关心具体的类是什么

```python
class Dog:
    def speak(self):
        return "汪汪"
class Cat:
    def speak(self):
        return "喵喵"
class Duck:
    def speak(self):
        return "嘎嘎"

# 统一的接口函数
def animal_sound(animal):
    print(animal.speak()) # 接收有speak()的对象

dog = Dog()
cat = Cat()
duck = Duck()
# 使用同一个函数接口处理不同对象
animal_sound(dog)   # 输出: 汪汪
animal_sound(cat)   # 输出: 喵喵
animal_sound(duck)  # 输出: 嘎嘎
```



## 文本处理

### re

re模块是python中用于正则表达式操作的标准库，提供了强大的模式匹配功能

**重要概念**：

正则表达式 : 指定了一组与之匹配的字符串的模式。

模式对象 : 通过 `re.compile()` 函数将正则表达式字符串编译后得到的对象。通常推荐先编译再使用，以提高效率。

匹配对象: 表示一次成功的匹配结果。包含匹配的起始位置、结束位置以及匹配到的子串等信息。

#### 正则表达式

指定了一组与之匹配的字符串的模式，由普通字符和元字符组成，元字符赋予了正则表达式强大的模式匹配能力

| 表示次数的元字符 | 含义             |
| ---------------- | ---------------- |
| `*`              | 匹配 0 次或多次  |
| `+`              | 匹配 1 次或多次  |
| `?`              | 匹配 0 次或 1 次 |
| `{n}`            | 匹配 恰好 n 次   |
| `{n,}`           | 匹配 至少 n 次   |
| `{n,m}`          | 匹配 n 到 m 次   |

| 表示特定字符的元字符 | 含义                                            |
| -------------------- | ----------------------------------------------- |
| `.`                  | 匹配 除换行符外 的任意字符                      |
| `\d`                 | 匹配 数字 (等价于 `[0-9]`)                      |
| `\D`                 | 匹配 非数字 (等价于 `[^0-9]`)                   |
| `\w`                 | 匹配 字母、数字或下划线 (等价于 `[a-zA-Z0-9_]`) |
| `\W`                 | 匹配 非字母、数字或下划线                       |
| `\s`                 | 匹配 空白字符 (空格、制表符、换行符等)          |
| `\S`                 | 匹配 非空白字符                                 |
| `[]`                 | 匹配 字符集 中的任意一个字符                    |
| `[^]`                | 匹配 不在字符集 中的任意字符                    |

| 表示边界的元字符 | 含义              |
| ---------------- | ----------------- |
| `^`              | 匹配字符串的 开头 |
| `$`              | 匹配字符串的 结尾 |
| `\b`             | 匹配 单词的边界   |

#### 常用函数与方法

| 功能     | 模块函数                     | 对象方法                  | 描述                                                         |
| -------- | ---------------------------- | ------------------------- | ------------------------------------------------------------ |
| 搜索     | `re.search(pattern,string)`  | `pattern.search(string)`  | 扫描整个字符串，找到第一个匹配的位置，返回一个匹配对象 ，如果未找到则返回 None。 |
| 查找所有 | `re.findall(pattern,string)` | `pattern.findall(string)` | 找到所有不重叠的匹配，并以列表形式返回匹配到的字符串。       |
| 拆分     | `re.split(pattern,string)`   | `pattern.split(string)`   | 根据模式匹配到的分隔符对字符串进行拆分，返回一个列表。       |

## 代码美学

### 列表推导式

通过在"[ ]"中加入一段简洁的描述，来创建一个新的列表，可以大大提高代码的简洁性，提高代码编写效率

```python
num = [1, 2, 3, 4, 5]
'''[x的表达式 x的for循环 条件语句]'''
squ1 = [x * x for x in num] #结果: [1, 4, 9, 16, 25]
squ2 = [x * x for x in num if x%2 == 0 and x > 2] #结果: [16]
```

****使用嵌套循环可以对列表的子列表元素进行操作****

```python
ls = [[1, 2], [3, 4], [5, 6]]
fls = [i for sls in ls for i in sls] #结果: [1, 2, 3, 4, 5, 6]
```

利用条件表达式 (三元运算符)，可以对原列表中的每个元素进行独立判断，并获得对应的结果

```python
numbers = [1, 2, 3, 4, 5]
result = ['Even' if x % 2 == 0 else 'Odd' for x in numbers]
# 结果: ['Odd', 'Even', 'Odd', 'Even', 'Odd']
```



### Type Hint

用于提示变量的类型，提高代码可读性，不具有强制性

```python
# 变量：变量类型
age: int = 30
name: str
# def 函数(参数:参数类型)->返回值类型
def fun(a:int,b:int)->str:
    pass
```

联合类型**：**变量类型可以是指定类型之一

```python
from typing import Union
ele:Union[int,str]
#等价于 ele:int|str (python3.10及以上版本)
```

可选类型：特殊的联合类型，变量可以是指定类型，也可为None

```python
from typing import Optional
ele:Optional[str]
#等价于 ele:Union[str,None]
#等价于 ele:str|None (python3.10及以上版本)
```



## **进阶技巧**

### generator生成器(yield关键字)

生成器是一种特殊的迭代器 。与普通的函数不同，生成器函数不是一次性计算并返回一个完整的结果列表，而是在被请求时逐个地生成值。

yield是生成器函数与普通函数的最大区别，它在生成器函数中替代return

```python
def gen():
    for i in range(10):
        yield i
```

每次函数执行到yield时，函数会返回一个值，然后暂停函数进程，等到下一次调用next()时会继续执行

```python
genA=gen() #先创建生成器迭代器对象

temp = next(genA) #然后通过该对象调用next函数
print(temp) #输出：0

temp = next(genA)
print(temp) #输出：1
```



#### 生成器表达式

与前文所说的列表推导式类似，主要区别有以下几点：

|          | 生成器表达式 ( )       | 列表推导式 [ ]     |
| -------- | ---------------------- | ------------------ |
| 返回类型 | generator              | list               |
| 执行过程 | 只在需要时计算对应元素 | 立刻计算每个元素   |
| 内存使用 | 通常较低，只存储逻辑   | 较高，存储所有结果 |

```python
'''(x的表达式 x的for循环 条件语句)'''
nums = [1, 2, 3, 4]
gen=(n * n for n in nums)
print(next(gen)) #进行第一次运算，输出：1
print(next(gen)) #进行第二次运算，输出：4
print(next(gen)) #进行第三次运算，输出：9
```

