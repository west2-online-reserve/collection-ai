随机
`from random import randint`
end参数指定print结束后的字符
`print('*',end=' ')`

输出格式
`print("%s's score is %d"%('Mike',87))`

>以下数值会被认为是False：
>- **为0的数字**，包括0，0.0
>- **空字符串**，包括''，""
>- 表示空值的 **None**
>- **空集合**，包括()，[]，{}

******
### 命令行常用命令

第一个常用的命令是：
`**dir** （windows环境下）`
显示出当前目录下的文件和文件夹。
```
cd 目录名
```

通过 dir 或 ls 了解当前目录的结构之后，可以通过“cd 目录名”的方式，进入到当前目录下的子目录里。

如果要跳回到上级目录，可以用命令：

```
cd ..
```

Windows下如果要写换盘符，需要输入

```
盘符:
```

比如从c盘切换到d盘

```
C:\Documents and Settings\Crossin> d:
```
通过使用 cd 命令进入到 py 文件保存的目录，再执行命令：

```
python 代码的文件名
```

就可以运行你写的程序了。

******
**google 是目前这个星球上最好用的搜索引擎**
*空格分割的1~3个关键词*
*直接使用报错信息搜索*
解决问题的网站：StackOverflow、SegmentFault
 [编程初学者如何使用搜索引擎](https://zhuanlan.zhihu.com/p/20683456)。
 
有空学习：- [如何将Python版「羊了个羊」打包成exe文件](https://mp.weixin.qq.com/s/bcJL3j55mmhi2XKbxbHWyQ)
```
`def sayhello(someone):
print(someone+' say hello')`

`if a==1:
    print()
elif a==2:
    print()
    else:`

### list
`l=[1,1,1,2,3]
print(l)`

`for i in l :
    print(i)`
```
可以是不同类型的组合
从0开始
```
`print(l[1])`
`l.append(1024)`
`del l[0]`
`ll=len(l)`
`l=[365,'',0.681]`
l[-1]表示最后一个元素
`l[1:3]`
左闭右开
`l[:3]`
字符串分割
`sentence='I am a student'
sentence.split()`
```
返回的是list
split默认是按照**空白字符**进行分割。

若想指定分割符号
`sentence.split('.')`

连接list
join是字符串的方法
`s = ';' li = ['apple', 'pear', 'orange'] fruit = s.join(li) print (fruit)`

**'apple;pear;orange'**
字符串**不能**通过索引访问去更改其中的字符。
******
### 读写文件
#### 读
`f=open('文件名')`
`data=f.read()`
`f.close`

other method:
`readline()#一行`
`readlines() #把内容按行读取至一个list中`

#### 写
`f = open('output.txt', 'w')`

`f.write('a string you want to write')`
******
### 处理文件中的数据

`for line inlines:`
`data=line.split()`

```
sum = 0 score_list = data[1:] # 学生各门课的成绩列表 for score in score_list: sum += int(score) result = '%s\t: %d\n' % (data[0], sum) # 名字和总分
```

```
f = open('scores.txt', encoding='gbk')
lines = f.readlines()
# print(lines)
f.close()

results = []

for line in lines:
    # print(line)
    data = line.split()
    # print(data)
    
    sum = 0
    score_list = data[1:]
    
    for score in score_list:
        sum += int(score)
    
    result = '%s \t: %d\n' % (data[0], sum)
    # print(result)
    results.append(result)

# print(results)

output = open('result.txt', 'w', encoding='gbk')
output.writelines(results)
output.close()
```

******
### 异常处理

```
try:
  f=open('non-exist.txt')
  print('File opened!')
except:
  print('File not exists.')
print('Done')
```

*******
### dictionary

```
d={key1:value1,key2:value2}
```

键/值对用冒号分割，每个对之间用逗号分割，整个字典包括在花括号中。

>关于字典的键要注意的是：
>
1.键必须是唯一的；
>
2.键只能是简单对象，比如字符串、整数、浮点数、bool值。
>
list就不能作为键，但是可以作为值。

ps : python字典中的键/值对没有顺序，我们无法用索引访问字典中的某一项，而是要用键来访问。

>两种方式：
>`print (score['段誉'])`
>`score.get('慕容复')`

第二种不会报错，但是第一种会报错
```
for name in score: print (score[name])
```

建立一个空的：
`d={}`
******
### 模块
`import random`
查询模块里面的方法：
`dir(random)`
```
from math import pi
```

```
from math import pi as math_pi
```
******
### 通过文件来记录数据
![[Pasted image 20251018205833.png]]

python有默认参数
******
###  查询天气

好玩的好玩的
模块 requests
发送网络请求，请求数据

```
import
requests
req=requests.get('http://www.baidu.com')
print(req)
req.encoding='utf8'
content=req.text
```

记事本中html保存html
**保存为HTML文件**​

- 点击"文件" → "另存为"
    
- 选择保存位置
    
- ​**关键步骤**​：在"文件名"处输入 `index.html`
    
- 在"保存类型"中选择"所有文件"
    
- 点击"保存"

******
解决文件格式修改失败的方法：
### Windows系统 - 显示文件扩展名（推荐）

​**步骤1：显示文件扩展名**​

1. 打开任意文件夹
    
2. 点击顶部菜单"查看"
    
3. 勾选"文件扩展名"选项
    

​**步骤2：重新保存文件**​

1. 现在你会看到文件的完整名称（如：`index.html.txt`）
    
2. 重命名文件，删除最后的 `.txt`，只保留 `index.html`
    
3. 系统会提示"如果改变文件扩展名，可能会导致文件不可用"，点击"是"

修改完就是一个网页啦~~~

运行一下看看能不能得到结果。如果提示编码的错误，试试在文件最开始加上：

```
# -*- coding: utf-8 -*-
```

![[Pasted image 20251018220347.png]]
*******
### 面向对象
```
class MyClass: pass mc = MyClass() print (mc)
```
```
class MyClass: name = 'Sam' def sayHi(self): print ('Hello %s' % self.name) mc = MyClass() print (mc.name) mc.name = 'Lily' mc.sayHi()
```

注意要有self在函数中
继承：
```
class Vehicle:
    def __init__(self, speed):
        self.speed = speed
    
    def drive(self, distance):
        print('need %f hour(s)' % (distance / self.speed))


class Bike(Vehicle):
    pass


class Car(Vehicle):
    def __init__(self, speed, fuel):
        Vehicle.__init__(self, speed)
        self.fuel = fuel
    
    def drive(self, distance):
        Vehicle.drive(self, distance)
        print('need %f fuels' % (distance * self.fuel))


# 测试代码
b = Bike(15.0)
c = Car(80.0, 0.012)

b.drive(100.0)
c.drive(100.0)
```

表达式从左往右运算，1和"heaven"做and的结果是"heaven"，再与"hell"做or的结果是"heaven"；0和"heaven"做and的结果是0，再与"hell"做or的结果是"hell"。

抛开绕人的and和or的逻辑，你只需记住，**在一个bool and a or b语句中，
当bool条件为真时，结果是a；当bool条件为假时，结果是b**。

有学过c/c++的同学应该会发现，这和bool?a:b表达式很像。

有了它，原本需要一个if-else语句表述的逻辑：

```
if a > 0:
    print ("big")
else:
    print ("small")
```

就可以直接写成：

```
print ((a > 0) and "big" or "small")
```

所以，and-or真正的技巧在于，确保a的值不会为假。最常用的方式是**使 a 成为 [a] 、 b 成为 [b]，然后使用返回值列表的第一个元素**：

```
a = ""
b = "hell"
c = (True and [a] or [b])[0]
print (c)
```

由于[a]是一个非空列表，**所以它决不会为假。即使a是0或者''或者其它假值，列表[a]也为真，因为它有一个元素**。
```
math.ceil(x) # 对x向上取整，比如x=1.2，返回2.0（py3返回2） 
math.floor(x) # 对x向下取整，比如x=1.2，返回1.0（py3返回1） 
math.pow(x,y) # 指数运算，得到x的y次方 
math.log(x) # 对数，默认基底为e。可以使用第二个参数，来改变对数的基底。比如math.log(100, 10) 
math.sqrt(x) # 平方根 math.fabs(x) # 绝对值
```
```
math.sin(x) math.cos(x) math.tan(x) math.asin(x) math.acos(x) math.atan(x)
```
注意：这里的x是以弧度为单位，所以计算角度的话，需要先换算

```
math.degrees(x) # 弧度转角度 math.radians(x) # 角度转弧度
```

### 正则表达式

正则表达式就是记录文本规则的代码，利用正则表达式来搜索文本
“\b”在正则表达式中表示单词的开头或结尾，空格、标点、换行都算是单词的分割。而“\b”自身又不会匹配任何字符，它代表的只是一个位置。所以单词前后的空格标点之类不会出现在结果里。
[]表示满足括号中任一字符。比如“[hi]”，它就不是匹配“hi”了，而是匹配“h”或者“i”
如果把正则表达式改为“[Hh]i”，就可以既匹配“Hi”，又匹配“hi”了。

re是python里的正则表达式模块。findall是其中一个方法，用来按照提供的正则表达式，去匹配文本中的所有符合条件的字符串。返回结果是一个包含所有匹配的list。

“.”在正则表达式中表示除换行符以外的任意字符。
如果我们用“i.”去匹配，就会得到
['i,', 'ir', 'il', 'is', 'if']
“\S”，它表示的是不是空白符的任意字符。注意是大写字符S。

“ * ”则不是表示字符，而是表示数量：它表示前面的字符可以重复任意多次（包括0次）

" * "在匹配时，会匹配尽可能长的结果。如果你想让他匹配到最短的就停止，需要用“.*?”。如“I.*?e”
```
import re 
text = "Hi, I am Shirley Hilton. I am his wife." 
m=re.findall(r"hi", text) 
if m: print (m) else: print ('not match')
```
匹配数字，我们可以用
[0123456789]
由于它们是连续的字符，有一种简化的写法：[0-9]。
类似的还有[a-zA-Z]的用法。
还有另一种表示数字的方法：\d

表示任意长度的数字，就可以用[0-9]*或者\d*
*表示的任意长度包括0，也就是没有数字的空字符也会被匹配出来。一个与*类似的符号+，表示的则是1个或更长。

如果要限定长度，就用{}代替+，大括号里写上你想要的长度。比如11位的数字：
\d{11}
1\d{10}

\w - 匹配字母或数字或下划线或汉字（我试验下了，发现python 3.x版本可以匹配汉字，但2.x版本不可以）
\s - 匹配任意的空白符
^ - 匹配字符串的开始
$ - 匹配字符串的结束

\S其实就是\s的反义，任意不是空白符的字符。同理，还有：
\W - 匹配任意不是字母，数字，下划线，汉字的字符
\D - 匹配任意非数字的字符
\B - 匹配不是单词开头或结束的位置
[a]的反义是[^a]，表示除a以外的任意字符。[^abcd]就是除abcd以外的任意字符。

之前我们用过*、+、{}来表示字符的重复。其他重复的方式还有：
? - 重复零次或一次
{n,} - 重复n次或更多次
{n,m} - 重复n到m次
^\w{4,12}$
这个表示一段4到12位的字符，包括字母或数字或下划线或汉字，可以用来作为用户注册时检测用户名的规则。（但汉字在python2.x里面可能会有问题）

\d{15,18}
表示15到18位的数字，可以用来检测身份证号码

^1\d*x?
以1开头的一串数字，数字结尾有字母x，也可以没有。有的话就带上x。

"\d+\.\d+"可以匹配出123.456这样的结果

“|”相当于python中“or”的作用，它连接的两个表达式，只要满足其中之一，就会被算作匹配成功。

于是我们可以把()的情况单独分离出来：
\(0\d{2,3}\)\d{7,8}

其他情况：
0\d{2,3}[ -]?\d{7,8}

合并：
\(0\d{2,3}\)\d{7,8}|0\d{2,3}[ -]?\d{7,8}

注意的是不同条件之间的顺序。匹配时，会按照从左往右的顺序

除了randint，random模块中比较常用的方法还有：

```
random.random()
```

生成一个0到1之间的随机浮点数，包括0但不包括1，也就是[0.0, 1.0)。

```
random.uniform(a, b)
```

生成a、b之间的随机浮点数。不过与randint不同的是，a、b无需是整数，也不用考虑大小。

```
random.uniform(1.5, 3)
random.uniform(3, 1.5)
# 这两种参数都是可行的

random.uniform(1.5, 1.5)
# 永远得到1.5
```

```
random.choice(seq)
```

从序列中随机选取一个元素。seq需要是一个序列，比如list、元组、字符串。

```
random.choice([1, 2, 3, 5, 8, 13]) #list
random.choice('hello') #字符串
random.choice(['hello', 'world']) #字符串组成的list
random.choice((1, 2, 3)) #元组
```

```
random.randrange(start, stop, step)
```

生成一个从start到stop（不包括stop），间隔为step的一个随机数。start、stop、step都要为整数，且start<stop。

比如：

```
random.randrange(1, 9, 2)
```

就是从[1, 3, 5, 7]中随机选取一个。

start和step都可以不提供参数，默认是从0开始，间隔为1。但如果需要指定step，则必须指定start。

```
random.randrange(4) # [0, 1, 2, 3]
random.randrange(1, 4) # [1, 2, 3]

# 下面这两种方式在效果上等同
random.randrange(start, stop, step)
random.choice(range(start, stop, step))
```

```
random.sample(population, k)
```

从population序列中，随机获取k个元素，生成一个新序列。sample不改变原来序列。

```
random.shuffle(x)
```

把序列x中的元素顺序打乱。shuffle直接改变原有的序列。

***
```
time.time()
```

返回的就是从epoch到当前的秒数（不考虑闰秒）。这个值被称为unix时间戳。

于是我们可以用这个方法得到程序开始和结束所用的时间，进而算出运行的时间：

```
import time
starttime = time.time()
print ('start:%f' % starttime)
for i in range(10):
    print (i)
endtime = time.time()
print ('end:%f' % endtime)
print ('total time:%f' % (endtime-starttime))
```

顺便再说下time中的另一个很有用的方法：

```
time.sleep(secs)
```

它可以让程序暂停secs秒。例如：

```
import time
print (1)
time.sleep(3)
print (2)
```

*******
### Time
得到用户输入的一个值方法是 eval()：

```
value = eval(input())
```

或者，如果你只是需要一个整数值，也可以：

```
value = int(input())
```
**********
### 列表解析

对其中的每一个元素进行判断，若模取2的结果为0则添加至新列表中。

使用列表解析实现同样的效果：

```
list_1 = [1, 2, 3, 5, 8, 13, 22]
list_2 = [i for i in list_1 if i % 2 == 0]
print (list_2)
```
没有指定参数名的参数必须在所有指定参数名的参数前面，且参数不能重复

*********
### 参数传递

在变量前加上星号前缀（\*），调用时的参数会存储在一个 tuple（元组）对象中，赋值给形参。在函数内部，需要对参数进行处理时，只要对这个 tuple 类型的形参（这里是 args）进行操作就可以了。因此，函数在定义时并不需要指明参数个数，就可以处理任意参数个数的情况。

```
def calcSum(*args):
    sum = 0
    for i in args:
        sum += i
    print (sum)
```

调用：

```
calcSum(1,2,3)
calcSum(123,456)
calcSum()
```

输出：

```
6
579
0
```

tuple 是有序的，所以 args 中元素的顺序受到赋值时的影响

```
func(**kargs)
```

上次说的 func(\*args) 方式是把参数作为 tuple 传入函数内部。而 func(\*\*kargs) 则是把参数以键值对字典的形式传入。

```
def printAll(**kargs):
    for k in kargs:
        print (k, ':', kargs[k])
        
printAll(a=1, b=2, c=3)
printAll(x=4, y=5)
```

输出：

```
a : 1
c : 3
b : 2
y : 5
x : 4
```

Python 的函数调用方式非常灵活，前面所说的几种参数调用方式，可以混合在一起使用。看下面这个例子：

```
def func(x, y=5, *a, **b):
    print (x, y, a, b)
    
func(1)
func(1,2)
func(1,2,3)
func(1,2,3,4)
func(x=1)
func(x=1,y=1)
func(x=1,y=1,a=1)
func(x=1,y=1,a=1,b=1)
func(1,y=1)
func(1,2,3,4,a=1)
func(1,2,3,4,k=1,t=2,o=3)
```

输出：

```
1 5 () {}
1 2 () {}
1 2 (3,) {}
1 2 (3, 4) {}
1 5 () {}
1 1 () {}
1 1 () {'a': 1}
1 1 () {'a': 1, 'b': 1}
1 1 () {}
1 2 (3, 4) {'a': 1}
1 2 (3, 4) {'k': 1, 't': 2, 'o': 3}
```

在混合使用时，首先要注意函数的写法，必须遵守：

- 带有默认值的形参(arg=)须在无默认值的形参(arg)之后；
- 元组参数(*args)须在带有默认值的形参(arg=)之后；
- 字典参数(**kargs)须在元组参数(*args)之后。

可以省略某种类型的参数，但仍需保证此顺序规则。

调用时也需要遵守：

- 指定参数名称的参数要在无指定参数名称的参数之后；
- 不可以重复传递，即按顺序提供某参数之后，又指定名称传递。

而在函数被调用时，参数的传递过程为：

1. 按顺序把无指定参数的实参赋值给形参；
2. 把指定参数名称(arg=v)的实参赋值给对应的形参；
3. 将多余的无指定参数的实参打包成一个 tuple 传递给元组参数(*args)；
4. 将多余的指定参数名的实参打包成一个 dict 传递给字典参数(**kargs)。

******
### lambda
使用 lambda 表达式来实现：

```
sum = lambda a, b, c: a + b + c

print (sum(1, 2, 3))
print (sum(4, 5, 6))
```

lambda 表达式的语法格式为：
```
lambda 参数列表: 表达式
```
```
def fn(x):
    return lambda y: x + y
    
a = fn(2)
print (a(3))
```

输出：

```
5
```
***********
### 作用域

在 Python 的函数定义中，可以给变量名前加上 global 关键字，这样其作用域就不再局限在函数块中，而是全局的作用域。

通过 global 改写开始的例子：

```
def func():
    global x
    print ('X in the beginning of func(x): ', x)
    x = 2
    print ('X in the end of func(x): ', x)
    
x = 50
func()
print ('X after calling func(x): ', x)
```

输出：

X in the beginning of func(x):  50

X in the end of func(x):  2

X after calling func(x):  2

******
### map函数

```
lst_1 = [1,2,3,4,5,6]
def double_func(x):
    return x * 2
lst_2 = map(double_func, lst_1)
print(list(lst_2))
```
map 的第一个参数是一个函数，之后的参数是序列，可以是 list、tuple。

也可以写成：

```
lst_1 = (1,2,3,4,5,6)
lst_2 = map(lambda x: x * 2, lst_1)
print(list(lst_2))
```
*******
### reduce函数
使用 reduce 需要先通过 from functools import reduce 引入

如果用 reduce 函数，就可以写成：

```
from functools import reduce

lst = range(1, 101)
def add(x, y):
    return x + y
print (reduce(add, lst))
```

解释一下：

```
reduce(function, iterable[, initializer])
```

第一个参数是作用在序列上的方法，第二个参数是被作用的序列，这与 map 一致。另外有一个可选参数，是初始值。

function 需要是一个接收2个参数，并有返回值的函数。它会从序列 iterable 里从左到右依次取出元素，进行计算。每次计算的结果，会作为下次计算的第一个参数。

提供初始值 initializer 时，它会作为第一次计算的第一个参数。否则，就先计算序列中的前两个值。

如果把刚才的 lst 换成 [1,2,3,4,5]，那 reduce(add, lst) 就相当于 ((((1+2)+3)+4)+5)。

同样，可以用 lambda 函数：

```
from functools import reduce

reduce((lambda x, y: x + y), range(1, 101))
```
**********
### 多线程
python 里有一个 threading 模块，其中提供了一个函数：

```
threading.Thread(target=function, args=(), kwargs={})
```

function 是开发者定义的线程函数，

args 是传递给线程函数的参数，必须是tuple类型，

kwargs 是可选参数，字典类型。

调用 threading.Thread 之后，会创建一个新的线程，参数 target 指定线程将要运行的函数，args 和 kwargs 则指定函数的参数来执行 function 函数。

![[Pasted image 20251026225959.png]]

******
### generator

如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，从而节省大量的空间。在Python中，这种一边循环一边计算的机制，称为生成器：generator。
创建`L`和`g`的区别仅在于最外层的`[]`和`()`，`L`是一个list，而`g`是一个generator。
如果要一个一个打印出来，可以通过`next()`函数获得generator的下一个返回值：

```
g = (x * x for x in range(10))
>>> next(g)
0
>>> next(g)
1
>>> next(g)
4
>>> next(g)
9
>>> next(g)
16
>>> next(g)
25
>>> next(g)
36
>>> next(g)
49
>>> next(g)
64
>>> next(g)
81
>>> next(g)
```

当然，上面这种不断调用`next(g)`实在是太变态了，正确的方法是使用`for`循环，因为generator也是可迭代对象：
```
>>> g = (x * x for x in range(10))
>>> for n in g:
...     print(n)
```
要把`fib`函数变成generator函数，只需要把`print(b)`改为`yield b`就可以了.
如果一个函数定义中包含`yield`关键字，那么这个函数就不再是一个普通函数，而是一个generator函数，调用一个generator函数将返回一个generator.
变成generator的函数，在每次调用`next()`的时候执行，遇到`yield`语句返回，再次执行时从上次返回的`yield`语句处继续执行。
调用该generator函数时，首先要生成一个generator对象，然后用`next()`函数不断获得下一个返回值
遇到`yield`就中断，下次又继续执行。执行3次`yield`后，已经没有`yield`可以执行了，所以，第4次调用`next(o)`就报错。
调用generator函数会创建一个generator对象，多次调用generator函数会创建多个相互独立的generator。
正确的写法是创建一个generator对象，然后不断对这一个generator对象调用`next()`：

```
>>> g = odd()
>>> next(g)
step 1
1
>>> next(g)
step 2
3
>>> next(g)
step 3
5
```
但是用`for`循环调用generator时，发现拿不到generator的`return`语句的返回值。如果想要拿到返回值，必须捕获`StopIteration`错误，返回值包含在`StopIteration`的`value`中：

```
>>> g = fib(6)
>>> while True:
...     try:
...         x = next(g)
...         print('g:', x)
...     except StopIteration as e:
...         print('Generator return value:', e.value)
...         break
...
g: 1
g: 1
g: 2
g: 3
g: 5
g: 8
Generator return value: done
```
******
### 迭代器

可以直接作用于`for`循环的数据类型有以下几种：
一类是集合数据类型，如`list`、`tuple`、`dict`、`set`、`str`等；
一类是`generator`，包括生成器和带`yield`的generator function。
这些可以直接作用于`for`循环的对象统称为可迭代对象：`Iterable`
可以使用`isinstance()`判断一个对象是否是`Iterable`
生成器都是`Iterator`对象，但`list`、`dict`、`str`虽然是`Iterable`，却不是`Iterator`
把`list`、`dict`、`str`等`Iterable`变成`Iterator`可以使用`iter()`函数：

```
>>> isinstance(iter([]), Iterator)
True
>>> isinstance(iter('abc'), Iterator)
True
```

`Iterator`甚至可以表示一个无限大的数据流，例如全体自然数。而使用list是永远不可能存储全体自然数的。
`Iterator`对象可以被`next()`函数调用并不断返回下一个数据，直到没有数据时抛出`StopIteration`错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过`next()`函数实现按需计算下一个数据，所以`Iterator`的计算是惰性的，只有在需要返回下一个数据时它才会计算。