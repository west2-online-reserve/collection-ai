# Task 1

### 学习内容

Python基础语法、认识生成式AI



### 疑问解答

- 是否需要好的显卡？

学习人工智能有一个好的显卡会有一定帮助（主要还是搞ai绘画方便点吧），但是不是刚需，你完全可以使用[Colab](https://colab.research.google.com/)或者gpu云服务器来进行机器学习代码的编写和运行



### 考核内容

#### 1. 配置环境

- 魔(ti)法(zhi)的使用

- 安装IDE集成开发环境

初学者可以选择使用[PyCharm](https://www.jetbrains.com.cn/pycharm/), 可以[申请学生免费许可证](https://blog.jetbrains.com/zh-hans/blog/2022/08/24/2022-jetbrains-student-program/)以使用专业版

个人更建议使用[VSCode](https://code.visualstudio.com/)（主要是PyCharm的Jupyter notebook太过难用， 建议初学者可以多使用Jupyter来编写程序）

- 安装Conda环境

使用[miniconda](https://docs.anaconda.com/miniconda/)就可以了,当然如果你不喜欢conda这样会污染命令窗口，使用pyenv或venv也都是可以的

- 学习python虚拟环境的配置

初学者一定要注意这个问题，从一开始就养成好的习惯

#### 2. 推荐教程

- python基础：

1. Crossin编程教室 [Python 入门指南 (python666.cn)](https://python666.cn/cls/lesson/list/)
2. Python - 100天从新手到大师的前10节课 [jackfrued/Python-100-Days: Python - 100天从新手到大师 (github.com)](https://github.com/jackfrued/Python-100-Days)
3. Python官方文档 [3.10.7 Documentation (python.org)](https://docs.python.org/zh-cn/3/)
4. 菜鸟教程 [Python3 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-tutorial.html)
5. 廖雪峰的官⽅教程 [Python教程 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/1016959663602400)
6. B站上有大量的入门基础课程，大家可以自行探索，找到适合自己的是最好的
7. 如果你想要系统的学习的话，我强烈推荐来自UCB的神课[CS61A](https://csdiy.wiki/%E7%BC%96%E7%A8%8B%E5%85%A5%E9%97%A8/Python/CS61A/)（课程难度较大，建议可以先选择一个更为友好的入门编程课程入门）

- 生成式AI认识：

[李宏毅2024 B站](https://www.bilibili.com/video/BV1BJ4m1e7g8/?spm_id_from=333.337.search-card.all.click&vd_source=e3594664d709db7578f4b2e76329df18)

[李宏毅2024 油管](https://www.youtube.com/watch?v=AVIKFXLCPY8&t=1s)

#### 3. 检验学习内容

在你完成python基础的知识学习后，你需要确保你对以下知识点能正确回答（如果不能你仍可以通过b站视频以及网上文档的方式进行弥补）

- 数据结构List，Dict的使用
- Lambda匿名函数
- Decorator装饰器
- 类Class的使用，Magic Methods的使用
- re正则表达式的使用
- 列表推导式

你可以写一个文档详细解释这些内容以加深印象（可以使用markdown的形式编写）

在完成生成式AI认识的学习后，你可以写一个文档详细向我们介绍一下一个大型语言模型训练的基本步骤

#### 4. 完成作业

1. 输出九九乘法表

2. 输入⼀个字符串，判断字符串中是否含有"ol"这个⼦串，若有把所有的"ol"替换为"fzu"，最后把字符串倒序输出

3. 输入⼀个列表（list），列表中含有字符串和整数，删除其中的字符串元素，然后把剩下的整数升序排序，输出列表

4. 创建一个字典（dict），为字典添加几个键为学号，值为姓名元素，删除学号尾号为偶数的元素，输出字典

5. 创建一个函数，这个函数可以统计一个只有数字的列表中各个数字出现的次数，通过字典方式返回

6. 设计⼀个商品类，它具有的私有数据成员是商品序号、商品名、单价、总数量和剩余数量。具有的 公有成员函数是：初始化商品信息的构造函数__init__，显示商品信息的函数display，计算已售出 商品价值income，修改商品信息的函数setdata

7. 尝试用所学的知识写一个斗地主随机发牌程序，将每个人的发牌以及多的三张牌的结果分别输出到player1.txt，player2.txt，player3.txt，others.txt四个文件中，可以不要求牌的花色

8. 实现一个装饰器，在开始执行函数时输出该函数名称， 并在结束时输出函数的开始时间和结束时间以及运行时间

9. 用所学的知识写一个斗地主随机发牌程序，将每个人的发牌以及多的三张牌的结果分别按照从大到小的顺序输出到player1.txt，player2.txt，player3.txt，others.txt四个文件中

10. 写一个列表推导式，生成一个5*10的矩阵，矩阵内的所有值为1，再写一个列表推导式，把这个矩阵转置

11. 了解类的魔术方法(Magic Method)。创建类MyZoo，实现以下功能：

    - 具有字典anmials，动物名称作为key，动物数量作为value

    - 实例化对象的时候，输出"My Zoo!"

    - 创建对象的时候可以输入字典进行初始化,若无字典传入，则初始化anmials为空字典

      ```
      myzoooo = MyZoo({"pig":5,'dog':6}) 
      myzoooo = MyZoo() 
      ```

    - print(myzoooo) 输出 动物名称和数量

    - 比较两个对象是否相等时，只要动物种类一样，就判断相等：

      ```
      输入：
      myzoooo1 = MyZoo({'pig':1})
      myzoooo2 = MyZoo({'pig':5})
      print(myzoooo1 == myzoooo2)
      输出:
      My Zoo!
      My Zoo!
      True
      ```

    - len(myzoooo) 输出所有动物总数

12. 写一个正则表达式，用于验证用户密码，长度在6~18 之间，只能包含英文和数字

### 作业解答

```python
1. for i in range(1,10):
    for j in range (1,i+1):
        print("%d * %d = %d "%(i,j,i*j),end=' ')
    print("\n")
--------------------------------------------------------------------------
    
2.def findOL(str):
    return str.find('ol')

word = input('Please enter str:')

if findOL(word)!=-1:
    print('exist')
    word = word.replace('ol','fzu')[::-1]
    print(word)
else:
    print('do not find')
--------------------------------------------------------------------------
3.def newList(L):
    return sorted([i for i in L if isinstance(i,int) ])

l = [1,'',2,'kong',3,'sfsf',0,1,4,4,54,64,8,2,]
print(newList(l))
--------------------------------------------------------------------------
4.d = {102300235:'林怡杰',102300234:'卓俊炜',102300309:'陈禹帆',102300308:'陈尚斌'}

print(d)
d = {k:v for k,v in d.items() if k%2==1}
print(d)
--------------------------------------------------------------------------
5.def countNum(L):
    d = {}
    for k in sorted(L):
        d[k] = L.count(k)
    return d

l = [2,1,6,5,5,7,7,6,3,1]
print(countNum(l))
--------------------------------------------------------------------------
6.class good(object):
    def __init__(self,name,id,price,totalSum,left):
        self.__id = id
        self.__name = name
        self.__price = price
        self.__totalSum = totalSum
        self.__left = left
    def display(self):
        self.income = self.__price*(self.__totalSum-self.__left)
        print(f'商品名为{self.__name}，商品序列号为{self.__id},单价为{self.__price}，总量为{self.__totalSum}，剩余数量为{self.__left},已售出商品价值{self.income}')
    def sedata(self):
        print('请输入商品的新价格')
        price = int(input())
        self.__price = price

skin = good('神王',1,100,100,50)
skin.display()
skin.sedata()
--------------------------------------------------------------------------
7.# -*- coding: utf-8 -*-
from random import sample
def init_deck():
    pile = []
    for i in range(4):
        for k in range(2,11):
            pile.append(str(k))
        pile.extend(['A', 'J', 'Q', 'K'])
    pile.extend(['大王','小王'])
    return pile

def deal_cards(pile):
    #分发地主牌
    landlord_cards = sample(pile,3)
    for i in landlord_cards:
        pile.remove(i)
    with open('others.txt','w',encoding='utf-8') as other:
        other.write('地主牌为：'+','.join(landlord_cards))
    
    #分发农民牌
    for i in range(1,4):
        player_cards = sample(pile,17)
        with open(f'player{i}.txt','w',encoding='utf-8') as p:
            p.write('你拿到的牌为：'+','.join(player_cards))

pile = init_deck()
deal_cards(pile)

#player1.txt，player2.txt，player3.txt，others.txt
--------------------------------------------------------------------------
8.import functools,time

def decorate(func):
    @functools.wraps(func)
    def wrapper(*arg,**kw):
        print('函数的名称为'+func.__name__)
        start = time.time()
        result = func(*arg,**kw)
        end = time.time()
        print('开始时间: %f' % start)  
        print('结束时间: %f' % end)  
        print('所用的时间为: %f秒' % (end - start))  
        return result
    return wrapper

@decorate
def add(a, b):  
    print('a + b = %d' % (a + b))  
    return a + b  
print(add(3,5))
--------------------------------------------------------------------------

9.from random import sample
def init_deck():
    pile = []
    for i in range(4):
        for k in range(2,11):
            pile.append(str(k))
        pile.extend(['A', 'J', 'Q', 'K'])
    pile.extend(['大王','小王'])
    return pile

def deal_cards(pile):
    #分发地主牌
    landlord_cards = settle_cards(sample(pile,3))
    for i in landlord_cards:
        pile.remove(i)
    with open('others.txt','w',encoding='utf-8') as other:
        other.write('地主牌为：'+','.join(landlord_cards))
    
    #分发农民牌
    for i in range(1,4):
        player_cards = settle_cards(sample(pile,17))
        for card in player_cards:
            pile.remove(card)
        with open(f'player{i}.txt','w',encoding='utf-8') as p:
            p.write('你拿到的牌为：'+','.join(player_cards))

def settle_cards(pile):
    result = []
    order = ['大王', '小王', '2', 'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3']
    for i in order:
        if i in pile:
            result.extend([i] * pile.count(i))  
    return result

    
pile = init_deck()
deal_cards(pile)

--------------------------------------------------------------------------
10.l = [[1 for i in range(10)] for i in range(5)]
for i in l:
    print(i)
print()
l1 = [[l[j][k]for j in range(5)]for k in range(10)]
for i in l1:
    print(i)
--------------------------------------------------------------------------    
11.class MyZoo(object):
    def __init__(self,animals={}):
        print('My Zoo!')
        if animals:
            self.animals = animals
    def __eq__(self,other):
        if set( self.animals.keys())==set(other.animals.keys()):
            return True
        return False
    def __str__(self) -> str:
        if not self.animals:
            return '动物园没有动物'
        else:
            return ','.join(f'{k}:{v}' for k,v in self.animals.items())
    def __len__(self):
        return (sum(self.animals.values()))
                
myzoooo1 = MyZoo({'pig':1})
myzoooo2 = MyZoo({'pig':5})
print(myzoooo1 == myzoooo2)
print(len(myzoooo2))

```





### 作业要求

1. 不要抄袭
2. 遇到不会可以多使用搜索引擎，实在没有找到解决方法可以来群里提问
3. 不限制使用chatgpt等大语言模型工具，但你需要确保你了解模型生成的内容的每一个细节，最好你可以在使用模型的内容的部分注释上reference from chatgpt这样的内容
4. 你还需要学习基本的git的使用，所有考核都采用git的方式进行上传
5. 作业内容可能会进行更新



### Bonus

完成CS61A的学习，完成相应的作业和lab（不要畏惧全英文的学习，你完全可以使用各类翻译软件帮助你学习(包括gpt)）,如果能啃下这门课那么你的编程水平将会超过绝大多数毕业生



# python知识点

##### 输入输出

print() input()

print()括号里可以是变量 数字 计算式 文字或字符串得加引号 单引号和双引号都行

print('*',end=' ')可以改变print结束后的字符（默认为回车）

print可以接收多个字符串，用‘，’隔开 在打印时就会被空格隔开

**input()的输入得到的都是字符串，输入为数字时也是数字的字符串**

input可以显示一个字符串来提醒用户 如**name = input('please enter your name:')**



##### 变量

python的变量声明不需要类型 有整型 字符串 浮点数 布尔类型

整数较大时可以用_进行分割 如 100_0000_0000

对于太大或太小的浮点数 必须用e来表示10的n次方

变量名和其他语言差不多

1. 第一个字符必须是字母或者下划线_
2. 剩下的部分可以是字母、下划线_或数字0~9
3. 变量名称是对大小写敏感的，myname 和 myName 不是同一个变量。

##### 选择结构和循环结构

```python
  #if选择语句
if 条件:
    选择执行的语句#语句需要冒号和统一的缩进 一般用四格 在整个文件都得统一
else 条件:
    if都失败后执行的语句
python 也有elseif简写为
elif 条件:
    语句
    
    
  #三元选择符 and or 与c语言中的？：一样
a = "heaven"
b = "hell"
c = True and a or b
print (c)
d = False and a or b
print (d)
#但是有个坑 a必须为真 当a为''时只会输出b
a = ""
b = "hell"
c = (True and [a] or [b])[0]
print (c)
#可以这样避免    
#还有一种三元运算符
value_if_true if condition else value_if_false

  #除了if选择语句 还有match匹配语句    
score = 'B'
match score:
    case 'A':
        print('score is A.')
    case 'B':
        print('score is B.')
    case 'C':
        print('score is C.')
    case _: # _表示匹配到其他任何情况
        print('score is ???.')
#使用match语句时，我们依次用case xxx匹配，并且可以在最后（且仅能在最后）加一个case _表示“任意值”，代码较if ... elif ... else ...更易读。
   
  #match语句除了可以匹配简单的单个值外，还可以匹配多个值、匹配一定范围，并且把匹配后的值绑定到变量：

age = 15
match age:
    case x if x < 10:
        print(f'< 10 years old: {x}')
    case 10:
        print('10 years old.')
    case 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18:
        print('11~18 years old.')
    case 19:
        print('19 years old.')
    case _:
        print('not sure.')
#在上面这个示例中，第一个case x if x < 10表示当age < 10成立时匹配，且赋值给变量x，第二个case 10仅匹配单个值，第三个case 11|12|...|18能匹配多个值，用|分隔。可见，match语句的case匹配非常灵活。

  #匹配列表 match语句还可以匹配列表，功能非常强大。
args = ['gcc', 'hello.c', 'world.c']
# args = ['clean']
# args = ['gcc']

match args:
    # 如果仅出现gcc，报错:
    case ['gcc']:
        print('gcc: missing source file(s).')
    # 出现gcc，且至少指定了一个文件:
    case ['gcc', file1, *files]:
        print('gcc compile: ' + file1 + ', ' + ', '.join(files))
    # 仅出现clean:
    case ['clean']:
        print('clean')
    case _:
        print('invalid command.')
'''第一个case ['gcc']表示列表仅有'gcc'一个字符串，没有指定文件名，报错；

第二个case ['gcc', file1, *files]表示列表第一个字符串是'gcc'，第二个字符串绑定到变量file1，后面的任意个字符串绑定到*files（符号*的作用将在函数的参数中讲解），它实际上表示至少指定一个文件；

第三个case ['clean']表示列表仅有'clean'一个字符串；

最后一个case _表示其他所有情况。

可见，match语句的匹配规则非常灵活，可以写出非常简洁的代码。'''

  #循环语句 
while 条件:
    循环执行的语句
for i in range(1,101):
    for循环中执行的语句#range(1,101)表示从1开始到100
#在循环中也可以使用continue和break提前结束循环

```

##### 模块

引入模块的方法

random模块

 ```python
 from 模块名 import 方法名
 #如 from random import randint
 randint(a,b)#产生一个a到b之间的数 包括ab
 dir(random)#这样可以知道模块中有哪些函数和变量
 #为了避免冲突 还可以给引入的方法改名
 from math import pi as math_pi 
 random.random()#产生一个0到1之间的浮点数 但是不包括1
 random.uniform(a,b)#产生ab之间的随机浮点数 ab无需是整数 也不用考虑大小
 random.choice(seq)#seq必须是一个序列 如list元组字符串 从序列中随机选择一个元素
 random.randrange(start,stop,step)
 #生成一个从start到stop（不包括stop），间隔为step的随机数 
 #start和step都可以不提供参数 默认从0开始 间隔为1 
 #提供一个参数时 就是提供stop 提供两个参数时就是提供start和stop
 random.sample(population, k)
 #从population序列中，随机获取k个元素，生成一个新序列。sample不改变原来序列。
 random.shuffle(x)
 #把序列x中的元素顺序打乱。shuffle直接改变原有的序列。
 #产生随机数时需要一个真实的随机数。在python中，默认用系统时间作为seed。你也可以手动调用random.seed(x)来指定seed。
 ```

time 模块

```python
import time
starttime = time.time()
print ('start:%f' % starttime)
for i in range(10):
    print (i)
endtime = time.time()
print ('end:%f' % endtime)
print ('total time:%f' % (endtime-starttime))
#用time.time()来查看时间戳
#还可以用time.sleep(sec)来让程序休眠一下 减少短时间内的请求 提高成功率
```



##### 字符串

1. 转义字符 \  \\\表示斜杠本身 还可以用来换行 如果代码写长了可以用\接着

2. 字符串可以用单引号也可以用双引号包括 具体用哪个看你里面的内容 如果里面单双引号都有 那可以用三引号 也可以用斜杠来转义

3. 字符串的格式化 

   两个字符串相加可以直接print(str1+str2)

```python
如果想输出字符串加数字就得
print(str+str(num))或者
print('my age is %d' %num)#同c语言 也有%.2f %s 多了一个元组
print("%s's score is %d" %('666',100))
 # %x 替换内容为十六进制的整数
 # 想写%时 需要%%进行转义
result = '%s\t %d\n' %(data[0],sum)#不止在print中可以格式化

#另一种格式化字符串的方法是使用字符串的format()方法，它会用传入的参数依次替换字符串内的占位符{0}、{1}……
print('Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125))
#1:.1f表示输出的格式

#最后一种格式化字符串的方法是使用以f开头的字符串，称之为f-string，它和普通字符串不同之处在于，字符串如果包含{xxx}，就会以对应的变量替换：
r = 2.5
s = 3.14*r**2
print(f'The area of a circle with radiud {r} is {s:.2f}')
```

4. 字符串的分割

sentence.split() split()会按照空格分割 分割成列表

sentence.split('.')就按句号分割 可以指定分割的符号

分割时即使后面没字符 也会分割出一个空串

'aaa'.split('a')会分成四个空串

5. 遍历字符串 可以通过for...in来遍历
6. 索引 通过[]来访问 也可以是负的
7. 切片 和list一样可以用[:]
8. 连接 可以用join newword = ','.join(word)
9. 如果字符串内部有很多换行，用`\n`写在一行里不好阅读，为了简化，Python允许用`'''...'''`的格式表示多行内容：

   ```bash
   >>> print('''line1
   ... line2
   ... line3''')
   ```

10. 字符串的编码 

英文字母用`ascll`码即可 为了表达更多的字符 引入`Unicode `为了节省空间 用`UTF-8`

用`ord()`可以获取字符的整数表示 用`chr()`可以把编码转为字符

用`b‘字符串’`可以把`str`变为以字节为单位的`bytes`

`‘字符串’.encode('ascill/utf-8')`可以编码为指定的`bytes`

要把`bytes`变为`str`，就需要用`decode()`方法：`b'字符串'.decode('ascill')`

如果`bytes`中只有一小部分无效的字节，可以传入`errors='ignore'`忽略错误的字节：

```bash
b'\xe4\xb8\xad\xff'.decode('utf-8', errors='ignore')
```

11. len(字符串)可以计算字符 如果换成bytes就计算字节数
11. splitlines(keepends)按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。

##### 类型转换

```python
可以用
 a = 1
 print(a,type(a)) 来查看a的类型
python提供了一些方法对数值进行类型转换：
int(x)     #把x转换成整数
float(x)  #把x转换成浮点数
str(x)     #把x转换成字符串
bool(x)   #把x转换成bool值、
在python中，其他类型转成 bool 类型时，以下数值会被认为是False：
为0的数字，包括0，0.0
空字符串，包括''，""
表示空值的 None
空集合，包括()，[]，{}
其他的值都认为是True。
```

##### 函数

1. 函数定义  python中函数的定义只需 def(): 加换行就行注意缩进
2. 函数参数  可以有默认参数 但是如有两个参数 只有一个有默认值 必须是后面的有默认参数
3. 函数名其实就是指向一个函数对象的引用，完全可以把函数名赋给一个变量，相当于给这个函数起了一个“别名”：

   ```bash
   >>> a = abs # 变量a指向abs函数
   >>> a(-1) # 所以也可以通过a调用abs函数
   1
   ```

###### 参数传递

```python
def func(arg1, arg2):
    print (arg1, arg2)
#在调用时，可以根据形参的名称指定实参
func(arg2=3, arg1=7)
#没有提供足够的参数时，会用默认值做参数的值
def func(arg1=1, arg2=2, arg3=3):
    print (arg1, arg2, arg3)
func(2, 3, 4)
func(5, 6)
func(7)
#可以指定部分参数
func(arg2=8)
func(arg3=9, arg1=10)
#也可以混用
func(11, arg3=12)

  #另一种参数传递方式 可变参数
def func(*args)
#可以接受任意数量的参数
def calcSum(*args):
    sum = 0
    for i in args:
        sum += i
    print (sum)
calcSum(1,2,3)
calcSum(123,456)
calcSum()
#如果已经有了一个list或者tuple要调用 就在前面加个*
nums = [1, 2, 3]
calcusm(*nums)
'''
在变量前加上星号前缀（*），调用时的参数会存储在一个 tuple（元组）对象中，赋值给形参。在函数内部，需要对参数进行处理时，只要对这个 tuple 类型的形参（这里是 args）进行操作就可以了。因此，函数在定义时并不需要指明参数个数，就可以处理任意参数个数的情况。
需要注意的是 tuple是有序的，所以args的元素顺序受赋值的影响
'''

  #最灵活的参数传递 关键字参数
func(**kargs)
def printAll(**kargs):
    for k in kargs:
        print (k, ':', kargs[k])
        
printAll(a=1, b=2, c=3)
printAll(x=4, y=5)
#这是把参数以键值对字典的形式传入
def printAll(**kargs):
    for k in kargs:
        print (k, ':', kargs[k])
        
printAll(a=1, b=2, c=3)
printAll(x=4, y=5)

#和可变参数类似，也可以先组装出一个dict，然后，把该dict转换为关键字参数传进去：
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, city=extra['city'], job=extra['job'])
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
#上面复杂的调用可以用简化的写法：

>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, **extra)
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
#**extra表示把extra这个dict的所有key-value用关键字参数传入到函数的**kw参数，kw将获得一个dict，注意kw获得的dict是extra的一份拷贝，对kw的改动不会影响到函数外的extra。

  #还能把几种方式混合在一起使用
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
'''
在混合使用时，首先要注意函数的写法，必须遵守：
带有默认值的形参(arg=)须在无默认值的形参(arg)之后；
元组参数(*args)须在带有默认值的形参(arg=)之后；
字典参数(**kargs)须在元组参数(*args)之后。

调用时也需要遵守：
指定参数名称的参数要在无指定参数名称的参数之后；
不可以重复传递，即按顺序提供某参数之后，又指定名称传递。
 

而在函数被调用时，参数的传递过程为：
按顺序把无指定参数的实参赋值给形参；
把指定参数名称(arg=v)的实参赋值给对应的形参；
将多余的无指定参数的实参打包成一个 tuple 传递给元组参数(*args)；
将多余的指定参数名的实参打包成一个 dict 传递给字典参数(**kargs)。
'''

```

###### 命名关键字参数

对于关键字参数，函数的调用者可以传入任意不受限制的关键字参数。至于到底传入了哪些，就需要在函数内部通过`kw`检查。

仍以`person()`函数为例，我们希望检查是否有`city`和`job`参数：

```python
def person(name, age, **kw):
    if 'city' in kw:
        # 有city参数
        pass
    if 'job' in kw:
        # 有job参数
        pass
    print('name:', name, 'age:', age, 'other:', kw)
```

但是调用者仍可以传入不受限制的关键字参数：

```bash
>>> person('Jack', 24, city='Beijing', addr='Chaoyang', zipcode=123456)
```

如果要限制关键字参数的名字，就可以用命名关键字参数，例如，只接收`city`和`job`作为关键字参数。这种方式定义的函数如下：

```python
def person(name, age, *, city, job):
    print(name, age, city, job)
```

和关键字参数`**kw`不同，命名关键字参数需要一个特殊分隔符`*`，`*`后面的参数被视为命名关键字参数。

调用方式如下：

```bash
>>> person('Jack', 24, city='Beijing', job='Engineer')
Jack 24 Beijing Engineer
```

如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符`*`了：

```python
def person(name, age, *args, city, job):
    print(name, age, args, city, job)
```

命名关键字参数必须传入参数名，这和位置参数不同。如果没有传入参数名，调用将报错：

```bash
>>> person('Jack', 24, 'Beijing', 'Engineer')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: person() missing 2 required keyword-only arguments: 'city' and 'job'
```

由于调用时缺少参数名`city`和`job`，Python解释器把前两个参数视为位置参数，后两个参数传给`*args`，但缺少命名关键字参数导致报错。

命名关键字参数可以有缺省值，从而简化调用：

```python
def person(name, age, *, city='Beijing', job):
    print(name, age, city, job)
```

由于命名关键字参数`city`具有默认值，调用时，可不传入`city`参数：

```bash
>>> person('Jack', 24, job='Engineer')
Jack 24 Beijing Engineer
```

使用命名关键字参数时，要特别注意，如果没有可变参数，就必须加一个`*`作为特殊分隔符。如果缺少`*`，Python解释器将无法识别位置参数和命名关键字参数：

```python
def person(name, age, city, job):
    # 缺少 *，city和job被视为位置参数
    pass
```

###### 参数检查

让我们修改一下`my_abs`的定义，对参数类型做检查，只允许整数和浮点数类型的参数。数据类型检查可以用内置函数`isinstance()`实现：

```python
def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x
```

添加了参数检查后，如果传入错误的参数类型，函数就可以抛出一个错误：

```bash
>>> my_abs('A')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in my_abs
TypeError: bad operand type
```

###### 默认参数的坑

默认参数很有用，但使用不当，也会掉坑里。默认参数有个最大的坑，演示如下：

先定义一个函数，传入一个list，添加一个`END`再返回：

```python
def add_end(L=[]):
    L.append('END')
    return L
```

当你正常调用时，结果似乎不错：

```bash
>>> add_end([1, 2, 3])
[1, 2, 3, 'END']
>>> add_end(['x', 'y', 'z'])
['x', 'y', 'z', 'END']
```

当你使用默认参数调用时，一开始结果也是对的：

```bash
>>> add_end()
['END']
```

但是，再次调用`add_end()`时，结果就不对了：

```bash
>>> add_end()
['END', 'END']
>>> add_end()
['END', 'END', 'END']
```

很多初学者很疑惑，默认参数是`[]`，但是函数似乎每次都“记住了”上次添加了`'END'`后的list。

原因解释如下：

Python函数在定义的时候，默认参数`L`的值就被计算出来了，即`[]`，因为默认参数`L`也是一个变量，它指向对象`[]`，每次调用该函数，如果改变了`L`的内容，则下次调用时，默认参数的内容就变了，不再是函数定义时的`[]`了。

 特别注意

定义默认参数要牢记一点：默认参数必须指向不变对象！

要修改上面的例子，我们可以用`None`这个不变对象来实现：

```python
def add_end(L=None):
    if L is None:
        L = []
    L.append('END')
    return L
```

现在，无论调用多少次，都不会有问题：

```bash
>>> add_end()
['END']
>>> add_end()
['END']
```

为什么要设计`str`、`None`这样的不变对象呢？因为不变对象一旦创建，对象内部的数据就不能修改，这样就减少了由于修改数据导致的错误。此外，由于对象不变，多任务环境下同时读取对象不需要加锁，同时读一点问题都没有。我们在编写程序时，如果可以设计一个不变对象，那就尽量设计成不变对象。

###### map函数

map 是 Python 自带的内置函数，它的作用是把一个函数应用在一个（或多个）序列上，把列表中的每一项作为函数输入进行计算，再把计算的结果以列表的形式返回。

map 的第一个参数是一个函数，之后的参数是序列，可以是 list、tuple。

```python
lst_1 = [1,2,3,4,5,6]
def double_func(x):
    return x * 2
lst_2 = map(double_func, lst_1)
#可用lst_2 = map(lambda x: x * 2, lst_1)代替
print(list(lst_2))

#map函数可以对多个序列进行操作
lst_1 = [1,2,3,4,5,6]
lst_2 = [1,3,5,7,9,11]
lst_3 = map(lambda x, y: x + y, lst_1, lst_2)
print(list(lst_3))
#参数个数要与map中提供的序列组数相同 如果少了会用none补全

```

###### reduce函数

map 可以看作是把一个序列根据某种规则，映射到另一个序列。reduce 做的事情就是把一个序列根据某种规则，归纳为一个输出



求1累加到100的和。寻常的做法大概是这样：

```python
sum = 0
for i in range(1, 101):
    sum += i
print (sum)
```

 

如果用 reduce 函数，就可以写成：

```python
from functools import reduce

lst = range(1, 101)
def add(x, y):
    return x + y
print (reduce(add, lst))
#也可以用lambda化简 reduce((lambda x, y: x + y), range(1, 101))
```

 

解释一下：

```python
reduce(function, iterable[, initializer])
```

第一个参数是作用在序列上的方法，第二个参数是被作用的序列，这与 map 一致。另外有一个可选参数，是初始值。

function 需要是一个接收2个参数，并有返回值的函数。它会从序列 iterable 里从左到右依次取出元素，进行计算。每次计算的结果，会作为下次计算的第一个参数。

提供初始值 initializer 时，它会作为第一次计算的第一个参数。否则，就先计算序列中的前两个值。

##### list列表

[1, 2, 3, 4, 5, 6, 7, 8, 9] 

这样一个中括号中的序列就是 list（列表）。列表的元素不一定是数字 还可以是不同类型混合 甚至可以嵌套list

1. 创建   l = []创建空列表
2. 访问 用[]来访问  l[i] l[-1]是最后一个元素 l[-3]是倒数第三个元素
3. 修改 同访问 直接赋值就行
4. 添加 用 l.append(要添加的元素)
5.  获取list的元素个数 len(l)
6. **list切片** 

```python 
l[1:3]表示从 1到3-1这几个元素  同样计数从0开始
l[1:]不指定最后一个数 就到结尾
l[:3]不指定第一个数 就从第一个元素开始
l[:]不指定 就返回整个列表
#切片中的数同样可以是负数
#常用 l[::-1]来得到反向列表
```

7. join连接

```python
s = ';'
li = ['apple', 'pear', 'orange']
fruit = s.join(li)#字符的join功能
print (fruit)#得到'apple;pear;orange'。
#也可以直接用 ';'.join(li)/';'.join(['apple','pear','orange'])
#用于连接的字符也可以是空的
```

8. 列表解析

```python
#通过已有的列表生成新的列表
list_1 = [1, 2, 3, 5, 8, 13, 22]
list_2 = []
for i in list_1:
    if i % 2 == 0:
        list_2.append(i)
print (list_2)
#这是通过循环来写
list_1 = [1, 2, 3, 5, 8, 13, 22]
list_2 = [i for i in list_1 if i % 2 == 0]
print (list_2)
#这是列表解析
#进一步的，在构建新列表时，还可以对于取出的元素做操作。比如，对于原列表中的偶数项，取出后要除以2，则可以
[i / 2 for i in list_1 if i % 2 == 0] 
```

9. 列表插入 l.insert(1,'jack') 第一个是插入的位置索引 第二个是插入的值
10. 删除末尾元素 l.pop()  删除指定位置元素 l.pop(i)i是索引
11. 列表加列表可以直接用+，也可以用list1.extend(list2)

##### 文件读写

1. 打开文件 open('文件名') f=open('text.txt')
2. 若文件中有中文需要f=open('text.txt',encoding='gbk')
3. 读文件 data = f.read() 会把所有的内容读入一个字符串
4. lines = f.readlines() 会把文件内容的一行一行返回为一个列表
5. 写文件 要在打开文件时指定为写入 f = open('text.txt','w')
6. 写入数据 f.write('a string you want to write') 括号里也可以是字符串变量
7. f.writelines(results)  括号里是list  也是一行一行写入
8. 关闭文件 f.close()

##### 异常处理

```python
try:
    print(int('0.5'))
except:
    print('发生错误')
```

##### 字典

基本格式为{key，value}

d = {}建立空字典

d = {key1 : value1, key2 : value2}

1.键必须是唯一的；

2.键只能是简单对象，比如字符串、整数、浮点数、bool值。

list就不能作为键，但是可以作为值。（因为list会变 而其他不会变）

3. 访问 字典中的键值对没有顺序 不能[数字]来访问 因为字典是无序的 得[key]，如果键是字符串 得加'' 想知道一个key存不存在 可以用in来判断 

4. 用get访问可以避免不存在而报错 如score.get(c,0)后面是不存在则返回的值

5. 可以通过 for in来遍历 in的变量是键

6. 增加 给一个键赋值score['慕容复'] = 88

7. 删除 del  del score['萧峰']  还有score.pop('萧峰')
8. 对于不变对象来说，调用对象自身的任意方法，也不会改变该对象自身的内容。相反，这些方法会创建新的对象并返回，这样，就保证了不可变对象本身永远是不可变的。

##### 集合

set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。

要创建一个set，用`{x,y,z,...}`列出每个元素：

```bash
>>> s = {1, 2, 3}
>>> s
{1, 2, 3}
```

或者提供一个list作为输入集合：

```bash
>>> s = set([1, 2, 3])
>>> s
{1, 2, 3}
#虽然显示为1 2 3 但是不代表有顺序 只是告诉你有这三个元素
```

重复元素在set中自动被过滤：

```bash
>>> s = {1, 1, 2, 2, 3, 3}
>>> s
{1, 2, 3}
```

通过`add(key)`方法可以添加元素到set中，可以重复添加，但不会有效果：

```bash
>>> s.add(4)
>>> s
{1, 2, 3, 4}
>>> s.add(4)
>>> s
{1, 2, 3, 4}
```

通过`remove(key)`方法可以删除元素：

```bash
>>> s.remove(4)
>>> s
{1, 2, 3}
```

set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：

```bash
>>> s1 = {1, 2, 3}
>>> s2 = {2, 3, 4}
>>> s1 & s2
{2, 3}
>>> s1 | s2
{1, 2, 3, 4}
```

##### 面向对象

python是一种高度面向对象的语言 其中的所有东西都是对象

```python
s = 'how are you'
#s被赋值后就是一个字符串类型的对象
l = s.split()
#split是字符串的方法，这个方法返回一个list类型的对象
#l是一个list类型的对象
通过dir()方法可以查看一个类/变量的所有属性：
dir(s)
dir(list)#用print打印出来

```

创建一个类

```python
class MyClass:
    pass

mc = MyClass()
print (mc)
#class加类名创建一个类 pass代表空语句
#类名加圆括号可以创建一个类实例
```

##### 元组

元组也是一种序列和list类似 但是元组中的元素在创建后就不能被修改

索引 切片 遍历 和list相同

```python
#元组指向不能变 但是元组的对象的内容可以变
t = ('a', 'b', ['A', 'B'])
t[2][0] = 'X'
t[2][1] = 'Y'
print(t)
```



##### 数学运算

python的数学运算模块叫做math，再用之前，你需要

```python
import math
```

 

math包里有两个常量：

```python
math.pi # 圆周率π：3.141592...
math.e # 自然常数：2.718281...
```

 

数值运算：

```python
math.ceil(x)
# 对x向上取整，比如x=1.2，返回2.0（py3返回2）

math.floor(x)
# 对x向下取整，比如x=1.2，返回1.0（py3返回1）

math.pow(x,y)
# 指数运算，得到x的y次方

math.log(x)
# 对数，默认基底为e。可以使用第二个参数，来改变对数的基底。比如math.log(100, 10)

math.sqrt(x)
# 平方根

math.fabs(x)
# 绝对值
```

 

三角函数: 

```python
math.sin(x)
math.cos(x)
math.tan(x)
math.asin(x)
math.acos(x)
math.atan(x)
```

注意：这里的x是以弧度为单位，所以计算角度的话，需要先换算

 

角度和弧度互换: 

```python
math.degrees(x)
# 弧度转角度

math.radians(x)
# 角度转弧度
```

##### 正则表达式

正则表达式是记录文本规则的代码

\b在正则表达式中代表开头或结尾 空格标点换行都算是分割

[]表达满足括号内的任一字符

r代表raw 表示不对字符串进行转义

```python
>>> print ("\bhi")
hi
>>> print (r"\bhi")
\bhi
```

```python
re.findall(r"hi", text) 
```

re是python里的正则表达式模块。findall是其中一个方法，用来按照提供的正则表达式，去匹配文本中的所有符合条件的字符串。返回结果是一个包含所有匹配的list。

1. '.'表示除了换行符以外的任意字符  '\s'表示除空白符的任意字符
2. '*'表示任意数量连续字符（0个也符合） '. *'是贪婪匹配 匹配到最长 '. *? '是懒惰匹配 匹配到最短     如果是一个或更长可以用+
3. [0-9]表示0-9这几个连续的字符 同理[a-z] 数字还可以用\d来表示
4. {}可以指定长度 如\d{11}表示11位的数字

5. 我们已经了解了正则表达式中的一些特殊符号，如\b、\d、.、\S等等。这些具有特殊意义的专用字符被称作“元字符”。常用的元字符还有：

> \w - 匹配字母或数字或下划线或汉字（我试验下了，发现python 3.x版本可以匹配汉字，但2.x版本不可以）
>
> \s - 匹配任意的空白符
>
> ^ - 匹配字符串的开始
>
> $ - 匹配字符串的结束

6. \S其实就是\s的反义，任意不是空白符的字符。同理，还有:

>  \W - 匹配任意不是字母，数字，下划线，汉字的字符
>
> \D - 匹配任意非数字的字符
>
> \B - 匹配不是单词开头或结束的位置

7. 之前我们用过*、+、{}来表示字符的重复。其他重复的方式还有：

> ? - 重复零次或一次
>
> {n,} - 重复n次或更多次
>
> {n,m} - 重复n到m次

##### lambda表达式

lambda表达式可以看做一种匿名函数

```python
def sum(a, b, c):
    return a + b + c
    
print (sum(1, 2, 3))
print (sum(4, 5, 6))

#可以用lambda表达式来实现
sum = lambda a, b, c: a + b + c

print (sum(1, 2, 3))
print (sum(4, 5, 6))
```

lambda 表达式的语法格式为：

```python
lambda 参数列表: 表达式
```

```python
def fn(x):
    return lambda y: x + y
    
a = fn(2)#a = lambda y: 2 + y
print (a(3))
#输出为5
```

##### python内置函数

* abs()返回数的绝对值
* all（）列表里的所有值是否为真
* any（）列表里有值为真即可
* basestring（）用于判断是否为string或Unicode isinstance（‘hello’,basestring）
* bin()把数字转为二进制
* bool（）转为布尔类型
* callable()用来检查是一个对象是否可以调用

#  石头剪刀布

##### 我的方法

```python
from random import choice
class player (object):
    def __init__(self):
        self.gesture = ['石头','剪刀','布']
    def choose(self):
        return choice(self.gesture)
    
class user (object):
    def __init__(self):
        self.gesture = ['石头','剪刀','布']
    def choose(self):
        print('请选择要出的东西 :1.石头 2.剪刀 3.布（输入数字）：')
        n = int(input())
        return self.gesture[n-1]
    
class game(object):
    def start(self):
        p1 = user()
        p2 = player()
        p1 = p1.choose()
        p2 = p2.choose()
        print(f"你选择了{p1}")
        print(f"对方选择了{p2}")
        self.win(p1,p2)
    def win(self,str1,str2):
        if str1==str2:
            print('平局')
        elif str1=='石头' and str2=='剪刀':
            print('你赢了')
        elif str1=='剪刀' and str2=='布':
            print('你赢了')
        else:
            print('你输了')
g = game()
g.start()
#游戏这个类的属性 应该有两个用户的选择 然后输赢的结果
#我的方法还是纯过程式编程
```

##### 网上的方法

1. 创建游戏类 里面有两个用户的选择 然后输赢的结果

```python
import random
class Game:
    def __init__(self):
        self.computer_pick = self.get_computer_pick()
        self.user_pick = self.get_user_pick()
        self.result = self.get_result()
    def get_computer_pick(self):
        random_number = random.randint(1,3)
        option = {1:'rock',2:'paper',3:'scissors'}
        return option[random_number]
    def get_user_pick(self):
        while 1:
            user_pick = input("输入石头/剪子/布：")
            if user_pick in ('rock', 'paper', 'scissors'):
                  break
            else:
                print('错误的输入！请在rock, paper, scissors中选择')
        return user_pick
    def get_result(self):
        if self.user_pick ==self.computer_pick:
            return '平'
        elif self.user_pick == 'paper' and self.computer_pick == 'rock':
            return "赢"
        elif self.user_pick == 'rock' and self.computer_pick == 'scissors':
            return '赢'
        elif self.user_pick == 'scissors' and self.computer_pick == ' pick':
            return '赢'
        # 在所有其他情况下，用户都会输
        else:
            return '输'
        
    def print_result(self):
        print(f"Computer's pick: {self.computer_pick}")
        print(f'Your pick: {self.user_pick}')
        print(f'You {self.result}')

game = Game()
game.print_result()
```

##### 我新写的

```python
from random import randint
class game(object):
    def __init__(self) -> None:
        self.p1 = 0
        self.p2 = 0
        self.play_game()

    def get_player(self):
        while 1:
            str = input('输入你的选择（石头、剪刀、布）：')
            if str not in ['石头','剪刀','布']:
                str = input('错误的输入，请重新输入')
            else :
                break
        return str
    
    def get_computer(self):
        d={1:'石头',2:'剪刀',3:'布'}
        return d[randint(1,3)]
    
    def get_result(self, player, computer):  
        if player == computer:  
            return '平'  
        elif (player == '石头' and computer == '剪刀') or \
             (player == '剪刀' and computer == '布') or \
             (player == '布' and computer == '石头'):  
            self.p1 += 1  
            return '赢'  
        else:  
            self.p2 += 1  
            return '输'  


    def print_info(self):
        print(f'你出了{self.player},电脑出了{self.computer},{self.result}了')
        print(f'目前比分为{self.p1}:{self.p2}')
        return 

    def play_game(self):
         while True:  
            self.player = self.get_player()  
            self.computer = self.get_computer()  
            self.result = self.get_result(self.player,self.computer)  
            self.print_info()  
            n = input('是否继续(输入y或n)：')  
            if n == 'n':  
                break  

g = game()

```



