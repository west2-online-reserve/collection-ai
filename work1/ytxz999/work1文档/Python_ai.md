## List的使用

### 创建List
```python
games=[csgo,'valront','genshin impact']
```
访问List元素
	直接使用索引（从0开始）   
	 //*games[-1]是最后的元素*
	
### 增删查改
List中增加元素 ，添加到末尾
```python
games.append('王者荣耀')
```

##### List中插入元素
```python
games.insert(0,'王者荣耀')
```
*//既有元素将右移一格*

##### 删除元素
```python
del games[0]
//前提：知道索引
game1=games.pop()
game2=games.pop(1)
//删除末尾元素，但是game1获得了被删除的值
game3=games.remove('王者荣耀')
//不知道元素位置，只删除第一个
```
### 组织列表

##### 排序
sort()  ,按字母排序，顺序
sort(reverse=True)按字母表逆序

```python
games.sort()
```
reverse()反转当前列表的顺序
```python
games.reverse()
```
也可以使用sorted进行临时排序
```python
print(sorted(games))
```
##### 长度
返回变量长度
```python
len(games)
```

### 操作列表
##### 遍历列表
在for循环之后需要缩进
```python
for onegame in games:
```
###### range
*打印一系列数*
1~5（不包含5）
```python
for value in range(1,5)
	print(value)
```
*创建链表*
range(1,6,2)第三个参数可以设置步长
```python
numbers= list (range(1,6))
```
###### 处理数字链表
	min(list)
	max(list)
	sum(list)
	处理数字列表



##### 切片
**games [a : b]** 返回索引a~b-1的元素
###### 复制列表
使用切片来复制
```python
games2 = game[:]
```
##### 元组
将 列表的[ ] 换成 ( )
列表会变成元组，不可以被修改，但是可以访问

## Dict的使用
### Dic的创建与访问
相比于列表，dict的特点更像是前一个元素变成了索引，后一个元素为对应的值
两个元素称为键值对
前一个元素类似于索引用于访问
```python
alien={'color' : 'green' , 'points' : '5'}
print(alien['color'])
```
### 对字典的操作
##### 删除键值对
可以类比列表
```python
del alien['points']
```
##### 遍历键值对
使用for循环设置两个变量来遍历items( )
```python
alien={'color' : 'green' ,
	   'points' : '5'    ,
	   'length' : '185'  }
	   
for key, value in alien.items():
	print(key,value)
```
当不需要字典中的值或者键的时候可以使用keys()，values()
```python
	for a in alien.keys():
	print(key,value)
```
###### set()
防止生成重复元素，找出独一无二的元素

## 匿名函数
匿名函数利用lambda关键词来定义
lambda 传入参数 ： 函数体（一行代码）
用于临时构建一个函数，只用一次的场景
匿名函数的定义中，函数体只能写一行代码
用例：
```python
test_fuc(lambda x,y:x+y)
#不使用lambda
def add(x,y):
return x+y
test_fuc(add(x,y))
```
## Decorator装饰器
decorator是一种特殊的函数
用于修改或增强其他函数的功能
它可以在不修改原函数代码的情况下，通过在原函数的定义之前使用语法来对其进行修饰
方法传参：* args, ** kwargs
给装饰器加上可变长形参就可以接受任何方法需要的参数
以下为用例（来自csdn）
```python
import time  # 导入time模块  
  
"""  
方法传参：*args, **kwargs  
"""  
  
def index(a):  # 有参数  
    time.sleep(1)  
    print('------------这里是index函数,接受的参数是：{}'.format(a))  
  
  
def home():  # 无参数  
    time.sleep(1)  
    print('------------这里是home函数------------')  
  
  
# 装饰器  
def outer(function_name):  
    def inner(*args, **kwargs):  # 添加可变长形参  
        start_time = time.time()  
        function_name(*args, **kwargs)  # 传参  
        end_time = time.time()  
        # 输出时间差  
        print("{} 执行耗时：{}".format(function_name.__name__, end_time - start_time))  
        print("------------------------------------------------")  
  
    return inner  
  
  
index = outer(index)  
index(555)  
  
home = outer(home)  
home()
————————————————
版权声明：本文为CSDN博主「aobulaien001」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/aobulaien001/article/details/132773432
```
## Class 和Magic methods
### 类的创建与初始化
**创建类**格式为class 类名（继承的类）:       *(如果没有确定的继承的类可以填object,这是所有类最终都要继承的类)* 
**初始化**通过定义一个特殊的__init__方法，把属性绑上去
```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

```

*注：__init__ 前后分别有两个下划线
	第一个参数必须是self,因为self指向创建实例本身，但是实例初始化不需要传入self*
	
**限制访问**如果想要内部属性不被外部访问，可以把属性的名称前面加上两个下划线__，如果实例变量名是__开头，就变成了私有变量（private）
如果需要更改可以创建一个方法
```python
class Student(object):
    ...

    def set_score(self, score):
        self.__score = score

```
### 继承与多态
类可以直接继承，
最大的好处是子类获得了父类的全部功能
第二个好处需要我们对代码做一点改进
当子类和父类都存在相同的`run()`方法时，子类的`run()`覆盖了父类的`run()`，在代码运行的时候，总是会调用子类的`run()`。这样，我们就获得了继承的另一个好处：多态。
```python
class Animal(object):
    def run(self):
        print('Animal is running...')
        
class Dog(Animal):
    def run(self):
        print('Dog is running...')
    def eat(self):
        print('Eating meat...')

 class Cat(Animal):
    def run(self):
        print('Cat is running...')


```
### Magic Method
_ _ init _ _ 用于初始化初始，不做介绍
_ _ str _ _ 在输出类或者输出以字符串形式的类的时候默认输出类的地址
```python
def __str__(self):
	return f"Student:{self.name},Age:{self.age}"
```
_ _ lt _ _ 魔术方法，用于类与类之间的比较(不含等号，大于小于)，输入参数是两个类，返回true或者false
```python
def __lt__(self,other):
	return self.age>other.age
```
_ _ le _ _ 魔术方法，用于类与类之间的比较不含(等号，大于等于和小于等于)，输入参数是两个类，返回true或者false
```python
def __le__(self,other):
	return self.age>=other.age
```
_ _ eq _ _ 魔术方法，用于类与类之间是否相等，输入参数是两个类，返回true或者false
```python
def __eq__(self,other):
	return self.age == 
	other.age
```
## 正则表达式
正则表达式是用来匹配字符串的工具
使用之前要import re
\d可以用来匹配数字，\w可以用来匹配数字和字母
要匹配变长的字符，在正则表达式中，用`*`表示任意个字符（包括0个），用`+`表示至少一个字符，用`?`表示0个或1个字符，用`{n}`表示n个字符，用`{n,m}`表示n-m个字符
在字符串前面加上r，不需要考虑\转义的问题
- `[0-9a-zA-Z\_]`可以匹配一个数字、字母或者下划线；
- `[0-9a-zA-Z\_]+`可以匹配至少由一个数字、字母或者下划线组成的字符串，比如`'a100'`，`'0_Z'`，`'Py3000'`等等；
- `[a-zA-Z\_][0-9a-zA-Z\_]*`可以匹配由字母或下划线开头，后接任意个由一个数字、字母或者下划线组成的字符串，也就是Python合法的变量；
- `[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}`更精确地限制了变量的长度是1-20个字符（前面1个字符+后面最多19个字符）。
## 列表推导式，Type Hint
列表推导式用于定义一个列表，但是与常规的不同，在常规的for语句中，需要额外写循环，但是列表推导式只需要一行代码，让代码更加简洁美观
```python
# 传统 for 循环
squares = [] 
for i in range(1, 11):
	  squares.append(i * i)
print(squares) 
 # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
  # 列表推导式（简洁高效）
squares = [i * i for i in range(1, 11)] 
print(squares) # 结果相同
```
**Type Hint**
用于**显式标注变量、函数参数、函数返回值的类型**
格式语法在变量名或者参数后面加上   ： 类型
对于复杂的类型的标志，需要用typing导入对应的工具
```python

name: str = "Alice

def add(a: int, b: int) -> int:
    return a + b
```

## generator生成器
**生成器（Generator）** 是一种特殊的迭代器（Iterator），它不需要一次性生成所有元素并存储在内存中，而是**按需动态生成元素**，从而节省内存空间，尤其适合处理大量数据或无限序列。
类似vector中的迭代器
```python
# 生成器表达式：不生成具体元素，返回生成器对象
nums_generator = (x * 2 for x in range(5))
print(nums_generator)
```
yield和return类似,但是函数下次调用的时候从yield开始
以下是用生成器生成斐波那契数列
```python
def fibonacci(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a  # 返回当前值，并暂停函数
        a, b = b, a + b  # 下次迭代从这里继续
        count += 1

# 创建生成器对象
fib = fibonacci(5)

# 遍历生成器
for num in fib:
    print(num)  # 依次输出：0, 1, 1, 2, 3
```
## 大型语言模型
**第一阶段**：让机器自学做“文字接龙”
需要学习语言知识和世界知识（通过爬虫程序在网上抓取信息，自监督学习）
**第二阶段**：让人类教模型使用自己的能力
训练模型的过程指令微调，利用人类的指令做出正确的回应（数据标注），这种通过人工标注数据训练模型的方式称为监督式学习
第一阶段所获得的参数被用作第二阶段的初始参数
**第三阶段**：学习哪个方法更好，称为强化学习
优化输出的整体质量

总结：大模型的训练包括三个阶段：（1）预训练阶段：从大规模网络数据中学习语言规律，得到基础模型（foundation model）；（2）指令微调阶段：引入人类老师，通过问题和答案对模型进行引导和纠正。（3）人类反馈强化学习阶段：让模型通过与人互动并学习用户反馈，进一步优化表现。后两个阶段也被称为“对齐阶段”，目的是让模型的行为符合人类的偏好与需求。以上就是语言模型训练的完整流程。