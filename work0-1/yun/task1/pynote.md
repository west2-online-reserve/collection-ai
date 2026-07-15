# python学习笔记
## 一.列表
**1.** 基本格式：
列表名 = [元素1，元素2，元素3...]

__注意__:

- 所有元素放在[]内，元素之间用','隔开

- 元素的数据类型可以不相同

**2.** 列表是可迭代对象，可以for循环遍历取值

**3.** 列表的常见操作

i.添加元素
| 名称 | 用法 |说明 |
|:------:|:------:|:------|
|append()|li.append("要加入的内容")| 整体添加|
|extend()|li.extend("要加入的内容")| 分散添加|
|insert()|li.insert(插入位置，"要加入的内容")|在指定位置插入元素，原有元素会后移|

ii. 修改元素 

直接通过下标修改

iii.查找元素

* __in__：判断指定元素是否存在列表中，如果存在就返回True,不存在就返回False

```
lie = ['a','b','c']

print('a' in lie) #True
print('d' in lie) #False  
```
* __index__ : 返回指定数据所在位置的下标，如果查找的数据不存在就会报错
* __count__ :  统计指定数据在当前列表出现次数

iiii.删除元素
  
  * __del__ : 删除指定下标的元素   
  __del li[1]__ 
  * __pop__ ：删除指定下标的数据，python3版本默认删除最后一个元素，下标不能超出范围 
                                      
    **li.pop(num)**
  * **remove** :根据元素的值进行删除，没有指定元素就报错，默认删除第一个指定元素

    **li.remove(指定元素)**

iiiii.排序

* **sort**(key,reverse):将列表按特定顺序重新排列，默认从小到大
* **reverse**( ):将列表倒序
  
iiiiii.列表推导式

*  格式一：结果 = [表达式 for 变量 in 列表]

     **注意**:in后面不仅可以放列表，还可以放range()、可迭代对象

*  格式二：[表达式 for 变量 in 列表 if 条件]

iiiiiii.列表嵌套

含义：列表里又有一个列表

示例：

```py
li =[1,2,3,[4,5,6]]                 #有点类似二维数组
print(li[1],li[3],li[3][2],sep='\n') 
'''输出结果
2

[4, 5, 6]

6'''
```

## 二.元组

2.1
* 格式：元组名 = (元素1，元素2，元素3)
* 所有元素放在()内，元素之间用','隔开
* 元素的数据类型可以不相同
* **注意**：只有一个元素，末尾必须加上逗号，否则返回唯一值的数据类型
  
2.2 元组与列表的区别

a.元组只有一个元素末尾必须加','  , 列表不需要<br>
b.元组只支持查询，不支持增删<br>
c.元组也有下标<br>

2.3 元组的应用场景<br>
a.函数的参数和返回值<br>
b.格式化输出本质上是元组<br>
c.元组数据不可修改，保护数据安全
## 三.字典


3.1 基本格式 ：字典名 = {键1：值1,键2：值2...}

* 以键值对形式保存，键和值之间用':'隔开，每个键值对用','隔开
* 字典中的键具备唯一性，但是值可以重复。键名重复前面的值会被后面的值替代

3.2 字典常见操作

3.2.1 查看元素

* 变量名[键名] 
* 变量名.get(键名)

```
dic = {'name' : 'kobe' , 'age' : 41 }
print(dic[1]) #报错，字典没有下标
print(dic['age']) #41
print(dic['sex']) #报错，键名不存在
print(dic.get('name')) #kobe
print(dic.get('sex')) #None
print(dic.get('sex','不存在'))#不存在，如果没有这个键名，返回自定义值
```
3.2.2 修改元素&新增元素

* 变量名[键名] = 值

**注意** 键名存在就修改，没有就新增
```
dic['name'] = 'laoda'
dic['sex'] = 'man'
print(dic)#{'name': 'laoda', 'age': 41, 'sex': 'man'}
```

3.2.3 删除元素

a. del 字典名 删除整个字典<br>
b. del 字典名[键名]  删除指定键值对，不存在就报错<br>
c. dic.clear()  清空字典,但保留这个字典<br>
d. dic.pop(键名)  删除指定键值对，并返回被删除的值,不存在就报错<br>
e. dic.popitem() 默认删除最后一个键值对并以元组形式返回被删除的键值对<br>

3.2.4 len() 求长度
* len(字典名) 返回字典中键值对的数量
```
dic = {'name' : 'kobe' , 'age' : 41 }
print(len(dic)) #2
```
3.2.5 keys(): 返回所有键名(可迭代对象)
```
dic = {'name' : 'kobe' , 'age' : 41 }
print(dic.keys()) #dict_keys(['name', 'age'])
for i in dic.keys():
    print(i)#name
             #age
```
3.2.6 values(): 返回字典中包含的所有值
```
dic = {'name' : 'kobe' , 'age' : 41 }
print(dic.values()) #dict_values(['kobe', 41])
for i in dic.values():
    print(i)#kobe
             #41
```
3.2.7 items(): 返回字典中包含的所有键值对,以元组形式
```
for i in dic.items():
    print(i)#('name', 'kobe')
             #('age', 41)
```
3.2.8 字典的应用场景

a.存储结构化数据<br>
b.函数的参数传递<br>

## 四.集合
4.1 基本格式：集合名 = {元素1，元素2，元素3...}
* 所有元素放在{}内，元素之间用','隔开
* 元素的数据类型可以不相同
4.2 集合的特点

a.集合中的元素是无序的，不能通过下标访问集合中的元素，也不能修改<br>
```
s1 = {'a','b',3}
print(s[0]) #报错，集合没有下标
print(s) #{3, a, b}  #集合中的元素是无序的，每次输出的顺序可能不一样
s2 = {3,2,1}
print(s2) #{1, 2, 3} ,数字以升序输出
```
**集合无序性的实现涉及哈希表**

b.集合中的元素具有唯一性，重复元素会被自动去重<br>
```
s = {'a','b',3,'a'}
print(s) #{3, a, b}  
```
4.3 集合的常见操作

4.3.1 添加元素
* add()：添加单个元素
```
s = {1,2,3}
s.add(4)
print(s) #{1, 2, 3, 4}
#如果添加的元素已经存在，集合不会报错，同时不会进行任何操作
s.add(2)
print(s) #{1, 2, 3, 4}
#一次只能添加一个元素，不能添加多个元素
s.add(5,6) #报错
s.add((5,6)) #可以添加一个元组
print(s) #{1, 2, 3, 4, (5, 6)}
```
* update()：把传入元素拆分后添加到集合中
```
s = {1,2,3}
s.update('abc')
print(s) #{1, 2, 3, 'a', 'b', 'c'}
s.update([4,5])
print(s) #{1, 2, 3, 4, 5, 'a', 'b', 'c'}
s.update((6,7))
print(s) #{1, 2, 3, 4, 5, 6, 7, 'a', 'b', 'c'}
```
4.3.2 删除元素
* remove()：根据元素的值进行删除，没有指定元素就报错
* discard()：根据元素的值进行删除，没有指定元素不会报错
* pop()：随机删除并返回被删除的元素，集合为空时会报错
```
s = {1,2,3}
s.remove(2)
print(s) #{1, 3}  
s.remove(4) #报错，元素不存在
s.discard(4) #不会报错，元素不存在
s.pop() #删除最左边元素并返回被删除的元素
print(s) #{3}  #剩下一个元素
```
4.3.3 集合的交集、并集、差集
* 交集：返回两个集合中都存在的元素
```
s1 = {1,2,3}
s2 = {2,3,4}
print(s1 & s2) #{2, 3}
```
* 并集：返回两个集合中所有的元素，重复元素只保留一个
```
s1 = {1,2,3}
s2 = {2,3,4}
print(s1 | s2) #{1, 2, 3, 4}
```
* 差集：返回在第一个集合中存在但在第二个集合中不存在的元素
```
s1 = {1,2,3}
s2 = {2,3,4}
print(s1 - s2) #{1}
print(s2 - s1) #{4}
```

## 五. 类型转换

5.1 

5.1.1 int()：将其他数据类型转换为整数类型，字符串必须是**数字**组成，否则会报错

5.1.2 float()：将其他数据类型转换为浮点数类型，字符串必须是**数字**组成，否则会报错

5.1.3 str()：将其他数据类型转换为字符串类型，**任何数据类型**都可以转换为字符串类型
          
5.1.4 bool()：将其他数据类型转换为布尔类型，以下数据类型转换为False：0、0.0、''、[]、()、{}、None，其他数据类型转换为True

5.1.5 list()：将可迭代对象转换为列表类型，**字符串**会被拆分成单个字符，**元组和集合**会被拆分成单个元素，**字典**会取键名

5.1.6 tuple()：将可迭代对象转换为元组类型，**字符串**会被拆分成单个字符，**列表和集合**会被拆分成单个元素，**字典**会取键名

5.1.7 eval()：将字符串类型转换为对应的数据类型，字符串必须是**合法的表达式**，否则会报错

5.1.8 set()：先去重，将可迭代对象转换为集合类型，**字符串**会被拆分成单个字符，**列表和元组**会被拆分成单个元素，**字典**会取键名

## 六.深浅拷贝,可变类型和不可变类型

6.1 浅拷贝：创建一个新的对象，拷贝第一层数据，嵌套层会指向原对象的内存地址
```py
import copy                                      #引入copy模块
li = [1,2,3,[4,5,6]]

li2 = li.copy()                                #浅拷贝

print(li)                                     #[1, 2, 3, [4, 5, 6]]
print(li2)                                    #[1, 2, 3, [4, 5, 6]]

print(id(li))                                 #内存地址
print(id(li2))                        #内存地址，浅拷贝后内存地址不同

li2[3][0] = 100                         #修改嵌套层数据
print(li)                                   #[1, 2, 3, [100, 5, 6]]
print(li2)                                  #[1, 2, 3, [100, 5, 6]]
print(id(li[3])) 
print(id(li2[3]))                 #内存地址，浅拷贝后嵌套层内存地址相同
```
* 优点：拷贝速度快，占用空间小，拷贝效率高
* 缺点：拷贝后原对象和新对象不完全独立，修改新对象的嵌套层数据会影响原对象

6.2 深拷贝：创建一个新的对象，拷贝所有层数据，嵌套层会指向新对象的内存地址
```py
import copy                                      #引入copy模块
li = [1,2,3,[4,5,6]]

li2 = copy.deepcopy(li)                                #深拷贝

print(li)                                     #[1, 2, 3, [4, 5, 6]]
print(li2)                                    #[1, 2, 3, [4, 5, 6]]

print(id(li))                                 #内存地址
print(id(li2))                        #内存地址，浅拷贝后内存地址不同

li2[3][0] = 100                         #修改嵌套层数据
print(li)                                   #[1, 2, 3, [4, 5, 6]]
print(li2)                                  #[1, 2, 3, [100, 5, 6]]
print(id(li[3])) 
print(id(li2[3]))                #内存地址，深拷贝后嵌套层内存地址不同
```
* 优点：拷贝后原对象和新对象完全独立，修改新对象不会影响原对象
* 缺点：拷贝速度慢，占用空间大，拷贝效率低

6.3 可变类型

含义 ：数据内容可以修改的类型，列表、字典、集合都是可变类型

变量名指向内存地址，内存地址指向数据内容，修改数据内容内存地址不变
```py
li = [1,2,3]
print(id(li)) #内存地址
li[0] = 100
print(li) #[100, 2, 3]
print(id(li)) #内存地址，修改数据内容内存地址不变。字典,集合同理
```
6.4 不可变类型

含义 ：数据内容不可以修改的类型，整数、浮点数、字符串、元组都是不可变类型

变量名指向内存地址，内存地址指向数据内容，修改数据内容内存地址改变
```py
a = 10
print(id(a))
a = 100
print(id(a)) #内存地址，修改数据内容内存地址改变。字符串,元组同理
```

## 七.函数

**定义** ：具有特定功能的代码块，可以重复使用

**格式** ：
```
def 函数名(参数1，参数2...):        
 #函数名遵循标识符命名规则，不能与内置函数重名，参数可以没有
 
    函数体     

  #函数执行时要执行的代码块，至少要有一行代码，如果没有代码可以用pass占位                 
    return 返回值

  #函数执行后返回的结果，可以没有返回值，默认返回None
  #return 后面的代码不会被执行
```
7.1 函数参数

7.1.1 必备参数（位置参数）：调用函数时必须传入的参数，实参和形参一一对应
格式：
```
def function(参数1，参数2...):
    函数体
```
7.1.2 默认参数：调用函数时可以不传入的参数，实参和形参一一对应，**默认参数必须放在必备参数后面**
格式：
```
def function(必备参数1，必备参数2...，默认参数1=默认值1，默认参数2=默认值2...):
    函数体
```
7.1.3 可变参数：调用函数时可以传入任意数量的参数，实参和形参不一一对应，**可变参数必须放在默认参数后面**
格式：
```
def function(必备参数1，必备参数2...，默认参数1=默认值1，默认参数2=默认值2...，*可变参数):                        #可变参数前面加*，函数体内可变参数会被当成一个元组
     函数体
```
7.1.4 关键字参数：调用函数时通过参数名传入的参数，实参和形参不一一对应，**关键字参数必须放在可变参数后面**
格式：
```
def function(必备参数1，必备参数2...，默认参数1=默认值1，默认参数2=默认值2...，*可变参数，**关键字参数):             
#关键字参数前面加**，函数体内关键字参数会被当成一个字典
     函数体
```
7.2 函数嵌套
定义：在一个函数内部定义另一个函数，内层函数可以访问外层函数的变量，但外层函数不能访问内层函数的变量

7.3 作用域

7.3.1 局部变量：在函数内部定义的变量只能在函数内部使用，函数执行结束后局部变量会被销毁

7.3.2 全局变量：在函数外部定义的变量可以在函数内部使用，函数执行结束后全局变量不会被销毁

7.3.3 

a. global关键字：在函数内部使用global关键字声明的变量是全局变量。

b. nonlocal关键字：在函数内部使用nonlocal关键字声明的变量是外层函数的局部变量。

示例
```py
def outer():
    a = 10
    def inner():
        nonlocal a
        a = 100
    inner()
    print(a)         # nonlocal声明a是外层函数的局部变量，修改a的值会影响outer函数内a的值
outer()   #100
```

7.4 匿名函数

定义：没有函数名的函数，使用lambda关键字定义，**只能有一个表达式，不能有return语句**

格式：函数名 = lambda 形参1，形参2...:表达式(返回值)

调用：结果 = 函数名(实参1，实参2...)
``` py
add = lambda x,y:x+y
print(add(1,1)) #2
```

参数设置与普通函数一样，可以设置默认参数、可变参数、关键字参数

7.4.1 lambda结合if判断
```py
max = lambda x,y:x if x>y else y
print(max(1,2)) #2
``` 
优点：代码简洁，适合定义简单的函数

缺点：功能有限，适合定义简单的函数，不适合定义复杂的函数

7.5 内置函数

```py
#查看所有内置函数
import builtins
print(dir(builtins))
```

常用内置函数：
* abs()：返回数字的绝对值
* sum()：对可迭代对象中的元素进行求和,字符串和字典不适用
* max()：返回可迭代对象中的最大值,字符串和字典不适用
  
  ```py
  print(max(1,2,-3,key=abs)) #-3, key参数指定abs函数作为比较的依据，返回绝对值最大的元素
  ```
* min()：返回可迭代对象中的最小值,字符串和字典不适用
* len()：返回对象的长度或元素个数
* zip()：将可迭代对象中的元素打包成一个个元组，返回由这些元组组成的zip对象
```py
li =[1,2,3]
li2 = ['a','b']
print(zip(li,li2)) #<zip object at 0x0000021BC8B0>
for i in zip(li,li2):
    print(i) #(1, 'a')
             #(2, 'b')
             #如果可迭代对象的元素个数不一致，zip会以最短的可迭代对象为准进行打包         
#转换成列表             
print(list(zip(li,li2))) #[(1, 'a'), (2, 'b')]
```

* map()：将指定函数作用于可迭代对象的每个元素，返回由结果组成的map对象
```py
def func(x):
    return x**2
li = [1,2,3]
print(map(func,li)) #<map object at 0x0000021BC80>
for i in map(func,li):
    print(i) #1
             #4
             #9
#转换成列表             
print(list(map(func,li))) #[1, 4, 9]
```

* reduce()：将指定函数作用于可迭代对象的前两个元素，并将结果与下一个元素继续作用，直到所有元素被处理，返回最终结果
```py
#引入functools模块
from functools import reduce
def func(x,y):
    return x+y
li = [1,2,3,4]
print(reduce(func,li)) #10, 1+2=3, 3+3=6, 6+4=10
```

7.6  拆包

含义：对于函数中的多个返回值，去掉元组，列表或字典，直接获取里面数据的过程


```py
def func():
    return 1,2,3
#方法一:要求返回值的数量和变量的数量一致，且顺序一致
a,b,c = func()
print(a,b,c) #1 2 3
#方法二:使用*变量名接收多余的返回值，变量会被当成一个列表
a,*b = func()
print(a,b) #1 [2, 3]
```

7.7 递归

定义：在函数内部调用函数本身的过程

优点：代码简洁，适合解决分治问题

缺点：递归层数过多会导致栈溢出，效率较低


7.8 函数引用

定义：函数也是一种数据类型，可以被赋值给变量，变量可以调用函数
```py
def func():
    print('hello')
greet = func #将函数func的引用赋值给变量greet
greet() #调用变量greet，输出hello
```
7.9 闭包

定义：在一个函数内部定义另一个函数，内层函数可以访问外层函数的变量，并且外层函数返回内层函数的引用

```py
def outer(x):
    def inner(y):
        return x+y
    return inner
add = outer(10) #调用outer函数，传入实参10，返回inner函数的引用，并将x的值绑定为10
print(add(5)) #调用add函数，传入实参5，返回x+y的结果，即10+5=15
```
7.10 装饰器

定义：装饰器本质上是一个闭包函数，在不修改函数代码的前提下，给函数增加额外功能的函数

条件：

1. 不修改原程序或函数的代码
2. 不修改原程序或函数的调用方式

7.10.1 标准版装饰器
```py
def decorator(func): #定义装饰器函数，参数func是被装饰的函数
    def wrapper(*args,**kwargs): #定义包装函数，接收被装饰函数的参数
        print('这是装饰器增加的功能') #在调用被装饰函数前增加的功能
        result = func(*args,**kwargs) #调用被装饰函数，并将结果保存到result变量中
        print('这是装饰器增加的功能') #在调用被装饰函数后增加的功能
        return result #返回被装饰函数的结果
    return wrapper #返回包装函数的引用
def func(x):
    print('这是被装饰的函数') #被装饰函数的功能
    return x**2 #被装饰函数的返回值
func = decorator(func) #使用装饰器装饰函数，返回包装函数的引用，并将其赋值给func变量
print(func(5)) #调用被装饰函数，输出装饰器增加的功能、被装饰的函数和装饰器增加的功能，最后输出被装饰函数的返回值
```

7.10.2 语法糖版装饰器

语法糖：在不改变程序功能的前提下，使用更简洁的语法来实现相同功能的代码

语句：@装饰器函数名，放在被装饰函数的定义前面，等价于被装饰函数 = 装饰器函数(被装饰函数)
```py
def decorator(func): #定义装饰器函数，参数func是被装饰的函数
    def wrapper(*args,**kwargs): #定义包装函数，接收被装饰函数的参数
        #(接收的参数要与被装饰函数的参数一致)
        print('这是装饰器增加的功能') #在调用被装饰函数前增加的功能
        result = func(*args,**kwargs) #调用被装饰函数，并将结果保存到result变量中
        print('这是装饰器增加的功能') #在调用被装饰函数后增加的功能
        return result #返回被装饰函数的结果
    return wrapper #返回包装函数的引用
@decorator #使用@符号将装饰器应用于函数，等价于func = decorator(func)
def func(x):
    print('这是被装饰的函数') #被装饰函数的功能
    return x**2 #被装饰函数的返回值
print(func(5)) #调用被装饰函数，输出装饰器增加的功能、被装饰的函数和装饰器增加的功能，最后输出被装饰函数的返回值
```
7.10.3 多层装饰器
定义：一个函数被多个装饰器装饰的情况，装饰器的执行顺序是从内到外，离函数最近的装饰器先装饰，即先执行最内层的装饰器，再执行外层的装饰器
```py
def decorator1(func):
    def wrapper(*args,**kwargs):
        print('这是装饰器1增加的功能')
        result = func(*args,**kwargs)
        print('这是装饰器1增加的功能')
        return result
    return wrapper
def decorator2(func):
    def wrapper(*args,**kwargs):
        print('这是装饰器2增加的功能')
        result = func(*args,**kwargs)
        print('这是装饰器2增加的功能')
        return result
    return wrapper
@decorator1
@decorator2
def func(x):
    print('这是被装饰的函数') #被装饰函数的功能
    return x**2 #被装饰函数的返回值
print(func(5)) #调用被装饰函数，输出装饰器增加的功能、被装饰的函数和装饰器增加的功能，最后输出被装饰函数的返回值
```
## 八.异常

8.1 异常类别

* NameError：变量名错误，访问了未定义的变量

* SyntaxError：语法错误，代码不符合Python的语法规则

* TypeError：类型错误，操作或函数应用于错误类型的对象
  
* ValueError：值错误，传入了正确类型但不合适的值

* IndexError：索引错误，访问了序列中不存在的索引

* KeyError：键错误，访问了字典中不存在的键

* ZeroDivisionError：零除错误，除数为零时发生

* IndentationError：缩进错误，代码块的缩进不正确

8.2 抛出异常

步骤：

1.创建一个Exception('xxx')对象，括号内可以传入异常信息

2.使用raise关键字抛出异常对象

```py
def func(x):
    if x < 0:
        raise Exception('x不能为负数') #抛出异常对象
    return x**2
print(func(-1)) #报错，x不能为负数
```

8.3 捕获异常

定义：当程序发生异常时，使用try...except语句捕获异常并进行处理，避免程序崩溃

使用try...except语句捕获异常

```
try:
    #可能发生异常的代码块
except Exception as e:
    #处理异常的代码块，e是捕获到的异常对象
```

## 九.模块和包

9.1 模块分类：一个.py文件就是一个模块，模块中可以定义函数、变量、类等

9.1.1 内置模块：Python自带的模块，可以直接使用，无需安装

如：random模块、math模块、time模块等

9.1.2 第三方模块：需要通过pip安装的模块

下载第三方模块：在命令行输入pip install 模块名

9.1.3 自定义模块：自己创建的模块

* 命名：模块名应遵循Python命名规范，使用小写字母、数字和下划线组成，避免使用Python关键字和内置模块名

9.2 导入模块

9.2.1 import 模块名：导入整个模块，使用模块名.函数名调用模块中的函数

语法：

导入模块：
```py
import 模块名
```

调用模块中的函数,变量：
```py
模块名.函数名(实参)
模块名.变量名
```

9.2.2 from 模块名 import 函数名,变量名：从模块中导入指定函数，直接使用函数名调用模块中的函数

语法：
```py
from 模块名 import 函数名
```
调用模块中的函数,变量：
```py
函数名(实参)
变量名
```

9.2.3 from 模块名 import *：从模块中导入所有函数，直接使用函数名调用模块中的函数

语法：
```py
from 模块名 import *     
```

**注意** :不建议使用这种方式导入模块，因为可能会导致命名冲突，无法确定函数的来源

9.2.4 

as：给模块或函数起别名，使用别名调用模块或函数

语法：
```py
import 模块名 as 别名
from 模块名 import 函数名1 as 别名,函数名2 as 别名
```

9.3 内置全局变量

* \_\_name__：当前模块的名字，如果是主模块，值为'\_\_main__'，如果是被导入的模块，值为模块名

```py
if __name__ == '__main__':
 #判断当前模块是否是主模块，如果是主模块就执行if语句块中的代码，如果是被导入的模块就执行else语句块中的代码
    print('这是主模块')
else:
    print('这是被导入的模块')
```

9.4 包

概念：包是一个包含多个模块的文件夹，包中必须有一个\_\_init__.py文件，\_\_init__.py文件可以是空的，也可以包含包的初始化代码。

作用：包将有关联的模块组织在一起，方便管理和使用

9.4.1 import 包名.模块名：导入包中的模块，使用包名.模块名.函数名调用模块中的函数

**注意：**  
* 如果包中有\_\_init__.py文件，导入包时会执行\_\_init__.py文件中的代码，如果\_\_init__.py文件中没有代码，导入包时不会执行任何代码

9.4.2 \_\_all\_\_：本质上时一个列表，列表里面的元素就代表要导入的模块

```py
#包的__init__.py文件
__all__ = ['模块名1','模块名2'...] #当使用from 包名 import *导入包时，只会导入__all__列表中指定的模块，其他模块不会被导入
```

9.4.3 包可以嵌套包：包中可以包含子包，子包中可以包含模块，导入子包中的模块时需要使用包名.子包名.模块名.函数名调用模块中的函数

## 十.面向对象编程

10.1 面向对象与面向过程

面向对象：以对象为中心，强调数据和操作数据的函数的结合，适合解决复杂问题

面向过程：以过程为中心，强调函数的调用和执行，适合解决简单问题

10.2 类和对象

类：是对一类事物的抽象，定义了这类事物的属性和方法


对象：是类的实例，具有类的属性和方法，可以通过对象调用类的方法

先有类再有对象，类是对象的模板，对象是类的实例

10.2.1 类的三要素

1. 类名：遵循标识符命名规则，首字母大写，多个单词使用驼峰命名法
2. 属性：描述对象的特征，通常在\_\_init\_\_方法中定义，self.属性名 = 属性值
3. 方法：描述对象的行为，函数定义在类的内部，第一个参数必须是self，表示对象本身，可以通过self访问对象的属性和方法

10.2.2 定义类

基本格式：
```py
class 类名:
    变量名 = 变量值 #类属性，所有对象共享
    def __init__(self,参数1,参数2...):
        self.属性1 = 参数1
        self.属性2 = 参数2...
    def 方法1(self,参数1,参数2...):
        #方法体
    def 方法2(self,参数1,参数2...):
        #方法体
```

查看类的属性 ：类名.属性名

10.2.3 创建对象

创建对象的过程也叫实例化对象

格式：对象名 = 类名(实参1，实参2...)
```py
class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
person1 = Person('Alice',30) #创建对象person1，传入实参'Alice'和30，调用__init__方法，初始化对象的属性
print(person1.name) #Alice，访问对象person1的name属性
print(person1.age) #30，访问对象person1的age属性
```

10.2.4 实例方法和实例属性

1. 实例方法
 
    由对象调用，至少有一个参数self，方法体内可以访问对象的属性和方法
```py
class Washer:
    height =800 #类属性，所有对象共享
    def wash(self): 
#实例方法至少有一个参数self，当对象调用实例方法时，会自动传入对象本身的引用作为参数，传递到实例方法的self参数中
        print('洗衣服')
wa1 = Washer() #创建对象wa1
wa1.wash() #调用对象wa1的wash方法，输出洗衣服
```

2. 实例属性
   
     由对象调用，实例属性在\_\_init\_\_方法中定义，实例属性的值可以不同，每个对象都有自己的实例属性
```py
class Person:
    def __init__(self,name,age):
        self.name = name #实例属性name
        self.age = age #实例属性age
person1 = Person('Alice',30) #创建对象person1，传入实参'Alice'和30，调用__init__方法，初始化对象的属性
person2 = Person('Bob',25) #创建对象person2，传入实参'Bob'和25，调用__init__方法，初始化对象的属性
print(person1.name) #Alice，访问对象person1的name属性
print(person1.age) #30，访问对象person1的age属性
print(person2.name) #Bob，访问对象person2的name属性     
print(person2.age) #25，访问对象person2的age属性
```

10.2.5 构造函数 \_\_init\_\_()

定义：在创建对象时自动调用的方法，用于初始化对象的**属性**

10.2.6 析构函数 \_\_del\_\_()

删除对象的时候，解释器会默认调用\_\_del\_\_方法

```py
class Person :
    def __init__(self):
        print('这是__init__')
    def __del__(self):
        print('被销毁了')
p=Person()
del p #删除p这个对象
#正常运行时，不会调用__del__(),对象执行结束后，系统会自动调用__del__()
```

## 十一 面向对象的三大特性

面向对象的三大特性

1. 封装
2. 继承
3. 多态

### 11.1 封装

封装：指的是隐藏对象中一些不希望被外部所访问到的属性或方法

#### 11.1.1
* 隐藏属性/方法（私有权限）：**只允许在类的内部使用**，无法通过对象访问
* 语法：在属性名或者方法名前面加上两个下划线

```py
class Person:
     name = 'manba'#类属性
     __age =41     #隐藏属性
     def man(self):
        print(self.__age)          #可正常访问

pe=Person()
print(pe.name)     
print(pe.__age)    #报错，'Person' object has no attribute '__age'
#隐藏属性实际上是将名字修改为：_类名__属性名
print(pe._Person__age) #正常访问age属性
```
#### 11.1.2<br>
* 私有属性/方法 ：_属性名/方法名 声明私有属性/方法，如果定义在类中，外部可以使用，子类也可以继承，但是在另一个py文件中通过from xxx import *导入时，无法导入。

### 11.2 继承

* 让类和类之间的关系转变为父子关系，子类默认继承父类的属性和方法

#### 11.2.1 单继承

语法

class  子类名(父类名):

```py
class Person:
    def man(self):
        print("what can i say")
class Person1(Person): #Person 类的子类
    pass #占位符，代码类下面不写任何东西，会自动跳过，不会报错
laoda=Person1()
laoda.man() #what can i say
```
总结：子类可以继承父类的属性和方法。

#### 11.2.2 继承的传递(多重传递)

```py
class Father :
    def hhh(self):
        print('hhh')
class Son(Father):   #Father的子类
    pass
class Grandson(Son): #Son的子类
    pass
man = Grandson()
man.hhh()   #hhh
```
#### 11.2.3  多继承

一个子类可以拥有多个父类，并且具有所有父类的属性和方法

```py
class Father1:      #父类1
    def hhh(self):
        print("HHH")
class Father2:      #父类2
    def hhh(self):
        print("hhh")
    
class Son(Father1,Father2):   #子类
    pass
son = Son()
son.hhh()            #HHH,当不同父类存在相同名字的方法，采用就近原则
print(Son.__mro__)   #查看搜索顺序
```

多继承的弊端

* 容易引发冲突
* 导致代码复杂度增加


### 11.3 重写

#### 11.3.1 覆盖

```py
class Father :     
    def hhh(self):
        print('hhh')
class Son(Father):   #Father的子类
    def hhh(self):   #覆盖父类的方法
        print('xxx')
man = Son()
man.hhh()   #xxx
```

#### 11.3.2 扩展

继承父类的方法，子类也有自己的方法

1. 父类名.方法名(self)
```py
class Father :     
    def hhh(self):
        print('hhh')
class Son(Father):   #Father的子类
    def hhh(self):  
        Father.hhh(self)
        print('xxx')   #新增的功能
man = Son()
man.hhh()   #hhh
            #xxx
```
2. super().方法名() --建议使用
   
   **另外写法**:super(子类名,self).方法名()

   super在python中是特殊的类，super()是使用super类创建出来的对象，可以调用父类的方法
```py
class Father :     
    def hhh(self):
        print('hhh')
class Son(Father):   #Father的子类
    def hhh(self):  
        super().hhh()  #可以调用父类的方法
        print('xxx')   #新增的功能
man = Son()
man.hhh()   #hhh
            #xxx
```

#### 11.3.3    新式类

class A(object)      
 
新式类 ：继承了object类或者该类的子类都是新式类

object --python为所有对象提供的基类(顶级父类),提供了一些内置的属性和方法，可以使用dir()查看
```py
print(dir(object))
```
python3中如果一个类没有继承任何类，则默认继承object类，因此python3都是新式类

### 11.4 多态

指同一种行为具有不同的表现形式

 多态的前提

* 继承
* 重写

#### 11.4.1 多态性：一种调用方式，不同的执行结果
```py
class Animal(object):
    def shout(self):
        print("叫")
class Cat(Animal):
    def shout(self):
        print("喵")
class Dog(Animal):
    def shout(self):
        print("汪")
cat = Cat()
dog = Dog()
def test(obj):     #test函数传入不同的对象，执行不同对象的方法
    obj.shout()
test(cat)
test(dog)
```

### 11.5 静态方法

使用@staticmethod来进行修饰。静态方法没有self,cls参数限制

静态方法与类无关，可以被转换成函数使用
```py
class Person(object):
    @staticmethod
    def hhh():
        print("hhh")
#静态方法既可以通过对象访问，也可以使用类访问
Person.hhh()
pe = Person()
pe.hhh()
```
取消了不必要的参数传递，有利于减少不必要的内存占用和性能消耗

### 11.6 类方法

使用装饰器@classmethod来标识为类方法。对于类方法，第一个参数必须是类对象，一般以cls作为第一个参数

    class 类名：

        @classmethod
        def 方法名(cls,参数)：
            方法体
类方法内部可以访问类属性，或者调用其他类方法

```py
class Person(object):
    @classmethod
    def hhh(cls):   #cls代表类对象本身，类本质上是一个对象
        print(cls,":","hhh")
Person.hhh()        #调用类方法,<class '__main__.Person'> : hhh
```

当方法中需要使用到类对象（如访问私有类属性等），定义类方法

类方法一般配合类属性使用

总结：

* 实例方法：方法内部可以访问实例属性也可以访问类属性
* 静态方法：不需要访问实例属性和类属性时使用，不能访问实例属性
* 类方法：方法内部只能访问类属性，不能访问实例属性

## 十二. 单例模式

### 12.1 \_\_new__

作用：

1. 在内存中为对象分配空间
2. 返回对象的引用
   
一个对象实例化过程，首先执行__new__(),返回一个实例对象，然后再调用__init__()初始化对象
```py
class Person(object):
    def __new__(cls,*args,**kwargs):
        print("这是new方法")
        return super().__new__(cls)
    def __init__(self):#初始化
        print('这是init方法')
pe=Person()
                                    # 这是new方法
                                    # 这是init方法

```
### 12.2 单例模式

**定义:** 单例模式是一种保证某个类在程序运行期间只有一个实例存在，并提供一个全局访问点的设计模式。

* 优点：节省内存空间
* 弊端：多线程访问时容易引发线程安全问题

**方式：**

1. 通过@classmethod
2. 通过装饰器实现
3. 通过重写__new__()实现
4. 通过导入模块实现

#### 12.2.1通过重写__new__()实现

设计流程

1. 定义一个类属性，初始值为None,用来记录单例对象的引用
2. 重写__new__()
3. 进行判断，如果类属性是None,把__new__()返回的对象引用保存
4. 返回类属性中记录的对象引用
```py
class A(object):
     obj = None
     def __new__(cls,*args,**kwargs):
        print("这是__new__()")
        if cls.obj ==None:
            A.obj = super().__new__(cls)
        return A.obj
        
a1 = A()
a2 = A()            #相当于把a1的引用赋给a2
print(a1,a2)        #内存地址相同
```
#### 12.2.2 通过导入模块实现单例模式

from 模块  import 对象 as 别名1<br>
from 模块  import 对象 as 别名2

模块就是天然的单例模式

应用：

1. 音乐播放器
2. 游戏软件
3. 数据库配置


## 十三. 魔法方法

### 13.1 __doc__():类的描述信息
```py
class Person(object):
    """人的描述信息"""
    pass
print(Person.__doc__)
```
### 13.2 \_\_module__:表示当前操作对象所在的模块
### 13.3 \_\_class__:表示当前操作对象所在的类
### 13.4 \_\_str__():如果类中定义了此方法，在打印对象时，默认输出该对象的返回值

* **\_\_str__()返回值必须为字符串**

### 13.5 \_\_call__():使一个实例对象成为一个可调用对象

*  callable():判断一个对象是否是可调用对象
  
```py
class A(object) :
    def __call__(self,*args,**kwargs):#调用一个实例对象，实际上在调用它的__call__()方法
        print("这是可调用对象")
a=A()
a()
```

### 13.6__dict__:查看类或对象中的所有属性,以字典形式返回

## 十四. 文件

### 14.1 文件基本操作

1. 打开文件
2. 读写文件
3. 关闭文件

#### 14.1.1 打开文件对象方法

1. open(): 创建一个file对象，默认是以只读模式打开
```py 
f=open(文件名/文件路径,访问模式)
```
**访问模式:**
|模式|操作|文件不存在|是否覆盖|
|:------:|:------:|:------:|:------:|
|r|只读|报错|-|
|r+|可读可写|报错|是|
|w|只写|创建|是|
|w+|可读可写|创建|是|
|a|只写(追加)|创建|追加|
|a+|可读可写(追加)|创建|追加|

**注意** :以w打开文件会清空文件内容，没有文件就创建文件

2. read(n): n表示从文件中读取的数据的长度，没有n就默认一次性读取文件的所有内容
3. write():将指定内容写入文件
4. close()：关闭文件
   
#### 14.1.2 属性

文件名.name:返回要打开的文件的文件名，可以包含文件的具体路径

文件名.mode：返回文件的访问模式

文件名.closed:检查文件是否关闭，关闭就返回True

#### 14.1.3 读写

1. read(n)

    *  n表示从文件中读取的的数据的长度,没有传值或者负值默认读取所有数据
2. readline()
    
    * 一次只读取一行，执行完毕，将文件指针移到下一行
3. readlines()
    * 按照行的方式把文件内容一次性读取，返回的是一个列表，每一行的数据就是列表中的一个元素
4. write()

#### 14.1.4 文件定位操作

1. tell(): 显示文件指针当前位置
2. seek(offset,whence):移动文件指针到指定位置
   
   *  offset：偏移量，表示要移动的字节数
   *  whence：起始位置，表示移动字节的参考位置，默认是0。0表示文件开头，1表示当前位置，2表示文件结尾
   *  seek(0,0) :将指针移到开头
#### 14.1.5 with open

作用：代码块执行完毕，系统会自动调用close(),可以省略文件关闭步骤

语法 ：with open(文件名,访问方式) as 对象名:
             
             代码块
### 14.2 编码格式

windows默认字符编码为GBK.

在文件中读写汉字时
```py
with open('test.txt',"r",encoding='utf-8') as f:
        print(f.read()) #读写中文
print(f.closed)
```
### 14.3 目录常用操作

导入模块 ：

import os

1. 文件重命名 ：os.rename(旧名字，新名字)
2. 删除文件： os.remove(文件名)
3. 创建文件夹：os.mkdir(文件夹名)
4. 删除文件夹：os.rmdir(文件夹名)
5. 获取当前目录：os.getcwd()
6. 获取目录列表：os.listdir()
   
     *os.listdir('../') ：获取上级目录列表

## 十五.迭代器,生成器

### 15.1 可迭代对象

含义：可以通过for...in ...这类语句遍历取值读取的对象称为可迭代对象

可迭代对象：str、list、tuple、dict、set等

#### 15.1.1可迭代对象的条件

1. 对象实现了__iter__()方法
2. \_\_iter__()方法返回了迭代器对象

#### 15.1.2for 循环工作原理
1. 先通过__iter__()获取可迭代对象的迭代器
2. 对获取到的迭代器不断调用__next__()方法来获取下一个值并赋值给临时变量i
#### 15.1.3 isinstance(o,t):判断一个对象是否是可迭代对象或者是一个已知的数据类型 o:对象，t：数据类型

```py
from collections.abc import Iterable

print(isinstance(123,Iterable))#False
print(isinstance("123",Iterable))#True 字符串是可迭代对象
```

### 15.2 迭代器

迭代器：可以记住遍历位置的对象。

iter():获取可迭代对象的迭代器,调用对象的__iter__()方法，并把该方法的返回结果作为自己的返回值
next():调用对象的__next__(),一个个地取元素，取完元素后会引发一个异常
```py
li=[1,2,3]
li1=iter(li)
print(next(li1)) #1
print(next(li1)) #2
print(next(li1)) #3
print(next(li1)) #抛出异常StopIteration
li2=li.__iter__()#获取可迭代对象的迭代器
print(li2.__next__()) #1
print(li2.__next__()) #2
print(li2.__next__()) #3
print(li2.__next__()) #抛出异常StopIteration
```

总结：
* next()和__next__()都是以迭代器对象作为参数。
* 可迭代对象有__iter__()方法，没有__next__()方法。
* 迭代器对象同时拥有__iter__()方法和__next__()方法。

### 15.3 迭代器协议

对象必须提供一个next()方法,执行该方法就返回迭代中的下一项,或者StopIteration

### 15.4 自定义迭代器类

两个特性：\_\_iter__()方法和__next__()方法

示例：
```py
class Myiter (object):
    def __init__(self,n):
        self.num = 0
        self.end = n       #循环次数为n
    def __iter__(self):
        return self
    def __next__(self):
        if self.num < self.end :
            m = self.num
            self.num+=1
            return m
        else :
            raise StopIteration
for i in Myiter(5):         #循环5次
    print(i)                #0,1,2,3,4
```

### 15.5 生成器

#### 15.5.1 生成器表达式

结果 = (表达式 for 变量 in 列表)

#### 15.5.2 生成器函数

python中，使用了yield关键字的函数称之为生成器函数

yield的作用：
1. 类似return，将指定值或者多个值返回给调用者
2. yield语句一次返回一个结果，然后挂起函数，执行next(),再重新从挂起点继续往下执行
```py
def gen(n):
    a = 0
    while a<n:
        yield a
        a+=1
for i in gen(5):
    print(i)
```

### 15.6 可迭代对象、迭代器和生成器三者关系 

* 可迭代对象：可以通过for循环遍历的对象，包括了迭代器和生成器
* 迭代器：可以记住自己遍历位置的对象，可以使用next（）函数返回值，只能往前不能往后。
* 生成器：特殊的迭代器，是python提供的通过简便的方法写出迭代器的一种手段

## 16.线程、进程

### 16.1 多线程
#### 16.1.1
进程：操作系统进行资源分配的基本单位，每打开一个程序至少有一个进程

线程：cpu调度的基本单位，每一个进程都至少有一个线程

导入线程模块
```py
import threading
```
Thread线程类参数

target ：执行的任务名<br>
args：以元组的形式给任务传参<br>
kwargs：以字典的形式给任务传参<br>

```py
import threading
import time
def func1(x):
    print(f"这是线程{x}")
    time.sleep(3)
    print(f'线程{x}执行完了')
def func2(x):
    print(f"这是线程{x}")
    time.sleep(2)
    print(f'线程{x}执行完了')
if __name__=='__main__':
    t1 =threading.Thread(target=func1,args=(1,))#以元组的形式传参
    t2 =threading.Thread(target=func2,args=(2,))
    #后台线程daemon:必须放在start()前面，主线程执行结束，子线程立即结束
    t1.daemon=True
    t2.daemon=True  
    #开始子线程start()
    t1.start()
    t2.start()
    #阻塞线程join():必须放在start()后面，暂停主线程，等子线程执行结束，主线程才会继续执行
    t1.join()
    t2.join()
    time.sleep(2)
    #获取线程名
    print("t1的进程名:",t1.name)
    print('t2的进程名:',t2.name)
    #更改线程名
    t1.name='子线程一'
    t2.name='子线程二'
    print("t1的进程名:",t1.name)
    print('t2的进程名:',t2.name)
    print("主线程执行完了")
```

#### 16.1.2 线程之间的执行是无序的
#### 16.1.3 线程间共享资源
#### 16.1.4 资源竞争

```py
import threading
a=0
b=10000000
def add1():
    for i in range(b):
        global a
        a+=1
    print('第一次累加：',a)
def add2():
    for i in range(b):
        global a
        a+=1
    print('第二次累加：',a)
if __name__=='__main__':
    a1 = threading.Thread(target=add1)
    a2 = threading.Thread(target=add2)
    a1.start()
    a2.start()
```

#### 16.1.5 线程同步

主线程和创建的子线程间各自执行完自己的代码直至结束

#### 16.1.6 互斥锁

概念：对共享数据进行锁定，保证多个线程访问共享数据不会出现数据错误问题

* acquire():上锁
* release():释放锁

**注意**：两种方法必须成对出现，否则容易造成死锁

```py
import threading
lock =threading.Lock() #创建全局锁
a=0
b=10000000
def add1():
    lock.acquire()     #上锁
    for i in range(b):
        global a
        a+=1
    print('第一次累加：',a)
    lock.release()     #解锁
def add2():
    lock.acquire()     #上锁
    for i in range(b):
        global a
        a+=1
    print('第二次累加：',a)
    lock.release()     #解锁
if __name__=='__main__':
    a1 = threading.Thread(target=add1)
    a2 = threading.Thread(target=add2)
    a1.start()
    a2.start()
```

互斥锁的作用：保证同一时刻只有一个线程去操作共享数据。

互斥锁的缺点：影响代码执行效率

### 16.2进程 
#### 16.2.1 含义
进程 ：操作系统进行资源分配和调度的基本单位，是操作系统结构的基础，进程里可以创建多个线程。

#### 16.2.2 进程的状态

1. 就绪状态：运行的条件都已满足，正在等待cpu执行
2. 执行状态：cpu正在执行其功能
3. 等待(阻塞)状态：等待某些条件的满足          

#### 16.2.3 进程语法结构

multiprocessing模块提供了Process类代表进程对象

Process类参数
1. target：执行的目标任务名，即子进程要执行的任务
2. args:以元组形式传参
3. kwargs:以字典形式传参

常用方法：
1. start():开启子进程
2. is_alive():判断子进程是否存活，存活返回True,死亡返回False
3. join():阻塞主进程等待子进程执行结束

常用属性
1. name:当前进程的别名。
2. pid:当前进程的编号

```py
import multiprocessing
import os
def func1():
    #os.getpid()获取当前进程编号
    print(f"Process1 pid:{os.getpid()}")
def func2():
    print(f"Process2 pid:{os.getpid()}")
if __name__ =="__main__":
    p1 = multiprocessing.Process(target=func1,name='进程1')
    p2 = multiprocessing.Process(target=func2,name='进程2')
    #获取主进程编号pid
    print('这是主进程pid:',os.getpid())
    #获取主进程的父进程pid
    print('这是主进程的父进程pid:',os.getppid())
    #开启进程
    p1.start()
    p2.start()
    #查看子进程编号
    # print(p1.pid)
    # print(p2.pid)
    #查看子进程别名
    # print(p1.name)
    # print(p2.name)
```

#### 16.2.4 进程间不共享全局变量

#### 16.2.5 进程间的通信

导入 Queue模块

from queue import Queue

**q.put():** 放入数据<br>
**q.get():** 取出数据<br>
**q.empty():** 判断队列是否为空<br>
**q.qsize():** 返回当前队列包含的消息数量<br>
**q.full():** 判断进程是否已满<br>

```py
from queue import Queue
#初始化一个队列队形
q = Queue(3)   # 最多可以接收3条消息，空或者负值表示没有上限
q.put('第一条消息') # 载入消息
q.put('第二条消息')
q.put('第三条消息')
print(q.full())  #True
print(q.get())   #获取队列的一条消息,然后将其从队列中移除
print(q.get())
print(q.qsize()) #返回队列中消息的数量
print(q.get())
print(q.empty())#True
```

#### 16.2.6 进程操作队列
```py
import multiprocessing
q1=multiprocessing.Queue()
def func1(q):
    for i in range(5):
        q.put(i)
        print(f"{i}正在写入队列") 
def func2(q):
    while not q.empty():
        print(q.get())
if __name__ =='__main__':
    p1=multiprocessing.Process(target=func1,args=(q1,))        
    p2=multiprocessing.Process(target=func2,args=(q1,))
    p1.start()
    p1.join()
    p2.start()
```
### 16.3 进程池

#### 16.3.1 阻塞和非阻塞

阻塞:程序遇到阻塞操作就停在原地，并立即释放cpu资源<br>
非阻塞：没有I/O操作或者通过其他方法让程序即使遇到I/O操作，也不会停止，而去执行其他操作，尽可能多的占用cpu资源。
#### 16.3.2 同步和异步

同步调用：提交完任务后，就在原地等待，直到任务运行完毕，才继续执行下一行代码<br>
异步调用：提交完任务后，不会在原地等待，直接执行下一行代码

#### 16.3.3 进程池
异步apply_async：不用等待当前进程执行完毕，随时根据系统调度来进行进程切换
```py
import time
from multiprocessing import Pool

def square(n):
    print('在计算')
    time.sleep(2)
    return n**2

if __name__ =='__main__':
    p=Pool(3) #创建进程池，最大进程数为3
    list1=[]
    for i in range(6):
        #apply_async异步
        result = p.apply_async(square,args=(i,))#square 函数名
        list1.append(result)
    p.close() #关闭进程池，不再接收新的请求
    p.join()  #等待所有子进程完毕
    for i in list1:
        #使用get()获取结果
        print(i.get())
```
同步：apply同步阻塞，等待当前子进程执行完毕后，再执行下一个进程
```py
# 异步：不用等待当前进程执行完毕，随时根据系统调度来进行进程切换
import time
from multiprocessing import Pool

def square(n):
    print('在计算')
    time.sleep(2)
    return n**2

if __name__ =='__main__':
    p=Pool(1) #创建进程池，最大进程数为1
    list1=[]
    for i in range(6):
        #apply_async异步
        result = p.apply(square,args=(i,))#square 函数名
        list1.append(result)
    p.close() #关闭进程池，不再接收新的请求
    p.join()  #等待所有子进程完毕
    print(list1)
```

#### 16.3.4 进程池的通信

Pool创建进程池，需要使用multiprocessing.Manager()中的Queue()

进程间的通信：multiprocessing.Queue()

线程间的通信：queue.Queue()

manager 模块，用于数据共享,支持很多类型：如value，array，list等等

**队列实例化** q= Manager().Queue()
```py
import os
import multiprocessing

def wd(q):
    print(f'wd的pid:{os.getpid()},父进程pid:{os.getppid()}')
    for i in '123':
        print('正在写入', i)
        q.put(i)
    q.put(None)  # 发送哨兵，通知消费者结束

def rd(q):
    print(f'rd的pid:{os.getpid()},父进程pid:{os.getppid()}')
    while True:
        item = q.get()
        if item is None:
            break
        print("取出数据:", item)

if __name__ == '__main__':
    print("主进程的pid", os.getpid())
    manager = multiprocessing.Manager()
    q = manager.Queue()
    p = multiprocessing.Pool(1) #初始化进程池
    p.apply_async(wd, args=(q,))
    p.apply_async(rd, args=(q,))
    p.close()
    p.join()
```

### 16.4 协程

含义：单线程下的开发，又称为微线程



#### 16.4.1 greenlet 第三方模块

注意：greenlet属于手动切换，当遇到io操作时，程序会阻塞
#### 16.4.2 通过greenlet实现任务的切换

```py
from greenlet import greenlet
def task1():
    print('哈哈哈')
    t2.switch()         #切换到t2
    print('task1继续执行')
def task2():
    print('呵呵呵')
    t1.switch()         #切换到t1
if __name__=='__main__':
    t1=greenlet(task1)
    t2=greenlet(task2)
    t1.switch()
```

#### 16.4.3 gevent 模块

遇到io操作时，会自动切换，属于主动式切换。

* gevent.spawn(函数名):创建协程对象
* gevent.sleep():耗时操作,能自动切换协程对象
* gevent.join():阻塞，等待某个协程执行结束
* gevent.joinall():等待所有协程对象都执行结束，参数是一个协程对象列表

```py
import gevent
def task1():
    print('哈哈哈')
    gevent.sleep(3)
    print('呵呵呵')
def task2():
    print('嘻嘻嘻')
    gevent.sleep(2)
    print('hhh')
if __name__=='__main__':
    t1=gevent.spawn(task1)
    t2=gevent.spawn(task2)
    gevent.joinall([t1,t2])
    print("程序执行完毕")
```

#### 16.4.4 monkey补丁：用有在模块运行时替换的功能
```py
import gevent
from gevent import monkey
import time
monkey.patch_all()#将用到的time.sleep()替换为gevent.sleep()
#一定要放在打补丁的对象前面
def task1():
    print('哈哈哈')
    time.sleep(3)
    print('呵呵呵')
def task2():
    print('嘻嘻嘻')
    time.sleep(2)
    print('hhh')
if __name__=='__main__':
    t1=gevent.spawn(task1)
    t2=gevent.spawn(task2)
    gevent.joinall([t1,t2])
    print("程序执行完毕")
```

### 16.5 总结

1. 线程是cpu调度的基本单位，进程是资源分配的基本单位
2. 进程、线程和协程对比

|        |        | 
|:------:|:------:|
|进程|切换所需的资源最大，效率最低
|线程|切换所需的资源一般，效率一般
|协程|切换所需的资源最小，效率最高


3. 多线程适合IO密集型操作(如文件操作，爬虫)，多进程适合cpu密集型操作(科学计算，高清解码)
4. 进程、线程和协程都可以完成多任务，根据需要灵活切换

## 17.正则表达式（字符串处理工具）

需要导入re模块

### 17.1 re.match

re.match(pattern,string)

* pattern 匹配的正则表达式
* string 要匹配的字符串
* 匹配开头的字符  
* 成功匹配返回match对象，失败返回None
* 使用group()方法提取数据

```py
import re
res=re.match('你','你好')    
print(res.group())
```

### 17.2 匹配单个字符

|字符|功能|
|:------:|:-------:|
|.|匹配除\n以外任意一个字符|
|[]|匹配[]中列举的字符
|\d|匹配数字|
|\D|匹配非数字|
|\s|匹配空白|
|\S|匹配非空白|
|\w|匹配单词字符
|\W|匹配非单词字符

1. . 匹配任意字符
    ```py
    res = re.match('.e','hello')
    ```
2. [] 
    ```py
    res = re.match('[he]','hello')
    ```
    匹配 0-9
    [0-9]

    匹配 a-zA-Z

    [a-zA-Z] 列举所有大小写字母

3. \d 匹配数字
    ```py
    res = re.match('\d\d','22ss')
    ```
4. \s 匹配空格
5. \w 匹配单词字符,即a-z,A-Z,0-9,_,汉字

### 17.3 匹配多个字符

1. \* 匹配前一个字符出现0次或者无数次，即可有可无  

    res = re.match('\w*','hello')  
    

2. \+ 匹配前一个字符出现1次或者无数次，即至少一次

    res = re.match('\w+','hello')
3.  ？匹配前一个字符出现1次或者0次
 
    res = re.match('\w？','hello')
4.  {m} 匹配前一个字符出现m次

    res = re.match('\d{3}','211')


6.  {m,n} 匹配前一个字符出现从m次到n次

    **注意** 要满足m<n的条件

    res = re.match('\d{3,6}','211')

### 17.4 匹配开头和结尾

1. ^ :匹配字符串开头/表示对...取反,

    res = re.match('[^py]','python')   表示匹配除p,y之外的字符

2. $: 匹配以...结尾
 
    res = re.match('n$','python')  

### 17.5 匹配分组

1. | (或) 匹配左右任意一个表达式 

    res = re.match('abc|def','abcdef')
    
    **注意** 先匹配左边的表达式
2. () 将括号中字符作为一个分组

    res = re.match('\w*@(123|qq).com','123@163.com')

3. \num 匹配分组num匹配到的字符串   ---经常在匹配标签时使用

    res = re.match(r'<(\w*)>\w*</\1>','\<html>login\</html>')

    res = re.match(r'<(\w*)><(\w*)>\w*</\2></\1>','\<html>\<body>login<\/body><\/html>')

   **注意**： 编号从1开始，由外向内排序 

4. (?P<name>) 分组起别名
5. (?P=name)  引用别名为name分组匹配到的字符串

    res = re.match(r'<(?P<L1>\w*)><(?P<L2>\w*)>\w*</(?P=L2)></(?P=L1)>','\<html>\<body>login<\/body><\/html>')

应用示例：匹配网址

```py
import re
li = ['www.baidu.com','www.python.org','http.jd.cn','www.py.en']
for i in li:
    res =re.match(r'www(\.)\w*\1(com|cn|org)',i)
    if res :
        print(res.group())
    else :
        print(f'{i}这个网址有错误')
```

### 17.6

1. search():扫描整个字符串并返回第一个匹配成功的对象，失败返回None
2. findall():从头到尾匹配，找到所有匹配成功的数据,返回一个列表
3. split(pattern,string,maxsplit):按模式分割字符串，返回分割后的列表
* pattern :正则表达式，要替换的内容
* string：字符串
* maxsplit:指定最大分割次数

```py
import re
li='apple,banana,peach,pear'
res =re.split(',',li)
print(res)      #['apple', 'banana', 'peach', 'pear']
```

4. sub(pattern,repl,string,count,):替换字符串中所有匹配模式的字串
* pattern :正则表达式，要替换的内容
* repl： 新字符串
* string：字符串
* count ：指定替换次数

```py
res =re.sub('\d','n','4565') #将数字替换为n
```



5. subn():与sub类似，但返回一个元组（新字符串，替换次数）


### 17.7 贪婪与非贪婪

#### 17.7.1 贪婪匹配 (默认)：在满足匹配时，匹配尽可能长的字符串。
```py
res = re.match('em*','emmmmmmmmmmm')
print(res.group())#emmmmmmmmmmm
```
#### 17.7.2 非贪婪匹配：在满足匹配时，匹配尽可能短的字符串。使用？来表示非贪婪匹配
```py
import re
res = re.match('em+?','emmmmmmmmmmm')
print(res.group())#em
```

### 17.8 原生字符串

正则表达式中，匹配字符串中的字符\需要\\\\\\\ ,原生字符串用\\\\表示

## 18. 常见模块

### 18.1 os模块

#### 18.1.1 作用：负责与操作系统交互
#### 18.1.2 获取平台信息

1. os.name:指示正在使用的工作平台（返回操作系统类型）
    
    对于windows,返回nt，对于linux，返回posix

2. os.getenv(环境变量名称): 读取环境变量

    print(os.getenv('path'))

3. os.path.split() : 把目录名和文件名分离，以元组形式接收，第一个元素是目录路径，第二个元素是文件名
4. os.path.dirname():目录名
5. os.path.basename():文件名
```py
import os
li = os.path.split(r'I:\code\pycode\bibao.py')
print(li[0],li[1])      #I:\code\pycode bibao.py
print(os.path.dirname(r'I:\code\pycode\bibao.py')) #I:\code\pycode
print(os.path.basename(r'I:\code\pycode\bibao.py'))#bibao.py
#如果路径以/结尾，返回空值，如果以\结尾，会报错
```

6. os.path.exists():判断路径(文件或目录)是否存在，存在返回True,不存在返回False
7. os.path.isfile():判断文件是否存在
8. os.path.isdir():判断目录是否存在

```py
import os
print(os.path.isfile(r'I:\code\pycode\bibao.py'))#True
print(os.path.isfile(r'I:\code\pycode'))#False
print(os.path.isdir(r'I:\code\pycode\bibao.py'))#False
print(os.path.isdir(r'I:\code\pycode'))#True
```
9. os.path.abspath():获取当前路径下文件或路径的绝对路径
```py
import os
print(os.path.abspath(r'bibao.py'))#I:\code\pycode\bibao.py
```

10. os.path.isabs():判断是否是绝对路径
```py
import os
print(os.path.isabs(r'bibao.py'))#False
print(os.path.isabs(r'I:\code\pycode\bibao.py'))#True
```

### 18.2 sys模块
作用：负责程序与python解释器交互

1. sys.getdefaultencoding():获取系统默认编码格式
2. sys.path:获取环境变量路径，与解释器相关。以列表形式返回，第一项为当前所在的工作目录
3. sys.platform :获取操作系统名称
4. sys.version:获取python解释器版本

### 18.3 time模块

1. 时间戳(timestamp)
2. 格式化的时间字符串(format time)
3. 时间元组(struct_time)

#### 18.3.1 time.sleep():延时操作
#### 18.3.2 time.time():获取时间戳，返回的是浮点型
#### 18.3.3 time.localtime():将一个时间戳转换为当前时区的struct_time
```py 
import time
t=time.localtime(time.time())
print(t.tm_year)
```

#### 18.3.4 time.asctime() :获取系统当前时间，把struct_time换成固定的字符串表达式

#### 18.3.5 time.ctime()：获取系统当前时间，把时间戳转换成固定的字符串表达式
#### 18.3.6 time.strftime(格式化字符串,struct_time)：将struct_time转换成时间字符串
```py
import time
t=time.localtime(time.time())
print(time.strftime('%Y年%m月%d日',t))
```
#### 18.3.7 time.strptime(时间字符串,格式化字符串)：将时间字符串转换为struct_time

```py
import time
print(time.strptime('2026年2月26日','%Y年%m月%d日'))
```

### 18.4 logging 模块
1. 作用：用于记录日志信息
2. 日志作用：

   1. 程序调试
   2. 了解软件程序运行情况是否正常
   3. 软件程序运行故障分析与定位

#### 18.4.1 级别排序

CRITICAL>ERROR>WARNING>INFO>DEBUG>NOTEST

```py
import logging
logging.debug('我是debug')
logging.info('我是info')
logging.warning('我是warning')
logging.error('我是error')
logging.critical('我是critical')
#logging默认的level是warning，只会显示级别大等于warning的日志信息
```
#### 18.4.2 logging.basicConfig():配置root logger的参数
1. filename：指定日志文件的文件名。
2. filemode:文件的打开模式，默认是追加
3. level:指定日志显示级别，默认是warning
4. format:指定日志信息的输出格式
```py
import logging
logging.basicConfig(filename='log.log',filemode='a',level=logging.NOTSET,format='%(levelname)s:%(asctime)s\t%(message)s')
logging.debug('debug')
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critical')
```

### 18.5 random 模块
作用：用于实现各种分布的伪随机数生成器，可以根据不同的实数分布来随机生成值
#### 18.5.1 random.random():产生大于0小于1之间的小数
#### 18.5.2 random.uniform():产生指定范围的随机小数
#### 18.5.3 random.randint():产生指定范围的随机整数,闭区间
#### 18.5.4 random.randrange(start,stop,[step]):产生start，stop范围内的整数，包含start但不包含stop
```py
import random
print(random.random())
print(random.uniform(1,3))
print(random.randint(1,3))
print(random.randrange(2,5,2))
```

## 十九 字符串

### 19.1 字符串编码：encode()和decode()

```py
a='hello'
a1=a.encode() #编码
print(type(a1))#bytes
print("编码后:",a1)
a2=a1.decode()#解码
print("解码后:",a2)
```

### 19.2 字符串运算符

|操作符|描述|实例
|:------:|:------:|:------|
|+|字符串连接|a+b 输出结果:hellopython
|*|重复输出字符串|a*2 :hellohello
|[]|通过索引获取字符串的一部分|a[1]输出e
|[:]|截取字符串的一部分|a[1:4]输出ell

### 19.3 切片
语法：[开始位置:结束位置:步长]

左闭右开

```py
st='asdfghjkl'
print(st[0:4])#asdf
print(st[3:])#fghjkl
print(st[0:6:2])#adg
print(st[-1:])#l
print(st[:-1])#asdfghjk
print(st[-1::-1])#lkjhgfdsa
```

### 19.4 字符串常见操作

#### 19.4.1 find(string,start,end):在start和end范围中查找

用法：检测某个字符串是否包含在字符串中，存在就返回这个子字符串开始位置的下标，否则返回-1

#### 19.4.2 index():和find用法一样，没找到会报错。
#### 19.4.3 count():返回某个子字符串在整个字符串出现次数
#### 19.4.4 lower():大写转小写
#### 19.4.5 upper():小写转大写
#### 19.4.6 startswitch(string,start,end):在指定范围检查字符串是否以某个子字符串开头，是返回True,否返回False
#### 19.4.7 endswitch(string,start,end):在指定范围检查字符串是否以某个子字符串结尾，是返回True,否返回False
#### 19.4.8 replace(old,new,count):替换指定字符串
#### 19.4.9 split(sep,maxsplit):按指定分隔符分割字符串
#### 19.4.10 capitalize():第一个字母大写

## 二十 类型注释

### 20.1语法：变量名:类型

```py
class Student:
    def __init__(
        self,
        name: str,
        birthday,
        sex: str,
        course: list[str],
        score: dict[str, float],
        location: tuple[int, int],
    ):
        pass
#列表、字典和元组可写明元素的类型，元组要写明每个元素的类型

def add(x,y) -> int:   #在说明号前注释
    return x+y
```

### 20.2 Literal
```py
from typing import Literal
sex :Literal['male','female']#将sex的值限制在'male','female'
```

### 20.3 Iterable,Callable
```py
x:Callable[[int,int],str]   #输入是两个整型，输出字符串
#Iterable 可以替代可迭代对象类型
```
