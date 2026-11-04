# Python 基础笔记

## 1. Python 基础

### 1.1 数据类型和变量

#### （1）整数

- 十六进制前缀 0x，由 0-9、a-f 组成，例：0xff00、0xa5b4c3d2
- 数字中间可加 _ 分隔方便阅读，不影响值：10_000_000_000、0xa1b2_c3d4

#### （2）浮点数

- 大数/小数推荐科学计数法，e 代表 10 的幂：1.23e9 = 1.23 × 10⁹

#### （3）字符串

- 转义符 \：\' 单引号、\" 双引号、\n 换行、\t 制表符、\\ 反斜杠
- 原始字符串 r''：内部字符不转义
- 多行字符串 '''...'''，保留换行

#### （4）布尔值

只有 True 和 False 两个值。

#### （5）空值、变量、常量

- 空值：None（不是 0，也不是空字符串）
- 变量：动态类型，直接赋值即可，无需声明类型
- 常量约定：变量名全部大写（语法上仍可修改，只是编程规范）

### 1.2 字符串和编码

#### （1）字符编码

- ASCII：最早编码，仅英文字母、数字、符号，共 127 个字符
- Unicode：统一全球所有文字，Python 3 的 str 底层就是 Unicode
- UTF-8：Unicode 的可变长编码，用于存储和网络传输，节省空间

#### （2）字符串相关函数

- ord(字符)：获取字符的 Unicode 编码（整数）
- chr(编码值)：把整数编码转回字符

```python
>>> ord('A')
65
>>> ord('中')
20013
>>> chr(66)
'B'
>>> chr(25991)
'文'
```

Unicode 转义写法：\u 加十六进制编码，例如 \u4e2d\u6587 等价于 "中文"。

#### （3）str 与 bytes

- 'ABC' → str 字符串，Unicode，面向字符
- b'ABC' → bytes 字节串，面向字节，前缀 b

转换方法：

- str → bytes：s.encode('utf-8')
- bytes → str：b.decode('utf-8')

```python
>>> 'ABC'.encode('ascii')
b'ABC'
>>> '中文'.encode('utf-8')
b'\xe4\xb8\xad\xe6\x96\x87'
>>> b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')
'中文'
```

注意：含有中文的 str 无法用 ASCII 编码，会报 UnicodeEncodeError。解码时遇到无效字节可用 errors='ignore' 忽略，但会丢失数据。

#### （4）len() 的两种行为

- len(str)：统计字符个数
- len(bytes)：统计字节个数

#### （5）字符串格式化

### 1.3 使用 list 和 tuple

#### （1）list（列表）

list 是一种有序集合，可以随时添加和删除其中的元素，元素类型可以不同。支持索引，追加（append）,插入（insert）,删除（pop）,嵌套，排序（sort）。

#### （2）tuple（元组）

元组相当于"不能改的列表"，没有 append、insert 等修改方法，初始化后不能重新赋值。

**易错点**

- t = (1) 不是元组，只是整数 1（括号被当成数学小括号）
- 单元素元组必须加逗号消除歧义：t = (1,)
- 元组本身不可变；但如果元组里存了 list，列表内部的元素可以修改

### 1.4 条件判断

### 1.5 模式匹配

Python 3.10+ 新增的 match-case 模式匹配，比 C/C++ 的 switch-case 灵活得多：

- 支持守卫条件：case x if x < 10:
- 支持多值匹配：case 11 | 12 | 13:
- 支持列表、字典、对象解构匹配

### 1.6 循环

### 1.7 使用 dict 和 set

#### （1）dict（字典）

Python 内置字典（dict），其他语言中称为 map，使用键-值（key-value）存储，查找速度极快。

**核心规则**

- key 必须是不可变对象（字符串、数字、元组等）

```python
>>> d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
```

**常用操作**

- 新增 / 修改：d['Adam'] = 67
- 安全取值：d.get(key, 默认值)，key 不存在不会报错
- 删除：d.pop('Bob')

#### （2）set（集合）

set 和列表比较像，但无序且元素不能重复。可以添加（add）,移除（remove）,还有交集并集（即为&和｜）

#### （3）补充：不可变对象

字符串是不可变对象，调用方法不会修改原字符串，而是返回新字符串：

```python
>>> a = 'abc'
>>> a.replace('a', 'A')
'Abc'
>>> a
'abc'
```

a 仍然是 'abc'，replace 返回的是新字符串 'Abc'，原对象没有被修改。

## 2. 函数

### 2.1 调用函数

调用函数：函数名(参数)，按位置依次传入。

Python 内置函数可以直接调用，例如 abs(-10)、max(1, 2)。

```python
>>> abs(-10)
10
>>> max(1, 2)
2
```

函数调用会在调用栈中创建栈帧，返回后销毁。

### 2.2 定义函数

#### （1）空函数

用 pass 定义空函数，pass 本身不执行任何操作，仅作为占位符，保证语法合法、代码可以先运行，后续再补逻辑。

```python
def nop():
    pass
```

#### （2）返回值

Python 函数返回多个值，本质上返回的是一个 tuple（元组）。

### 2.3 函数的参数

顺序规则：位置参数 → 默认参数 → 可变参数 → 命名关键字参数 → 关键字参数

#### （1）位置参数

按顺序依次传入，一一对应。

#### （2）默认参数

- 规则：必选（位置）参数在前，默认参数在后。如果提供的参数数量多于位置参数，就按顺序给默认参数。
- 重大坑：默认参数必须指向不可变对象！

错误写法：

```python
# 错误写法
def add_end(L=[]):
    L.append('END')
    return L
# 多次调用会累加，因为列表对象在函数定义时就创建，复用同一个
```

正确写法，用不可变对象 None：

```python
# 正确写法，用不可变对象 None
def add_end(L=None):
    if L is None:
        L = []
    L.append('END')
    return L
```

不变对象优势：内容不可修改，减少出错；多线程并发读不需要加锁。

#### （3）可变参数 `*args`

- 格式：参数前加 *；函数内部接收为一个 tuple
- 调用时，可直接传多个值，也可以对 list / tuple 解包传入：`*my_list`。即Python允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去。

#### （4）关键字参数 `**kw`

- 格式：参数前加 **；函数内部接收为一个 dict
- 作用：传入任意带参数名的键值对
- 可对已有字典解包传入：`**extra`

注意：函数内的 kw 是原字典的拷贝，修改 kw 不会影响外面的字典

```python
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}

>>> person('Jack', 24, city=extra['city'], job=extra['job'])

name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
```

或者，可以用简化的写法：

```python
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}

>>> person('Jack', 24, **extra)

name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
```

#### （5）命名关键字参数

- 作用：限制可以传入的关键字参数名称
- 写法：用单独 * 作为分隔符，后面就是限定的参数名
- 例如，只接收city和job作为关键字参数。这种方式定义的函数如下：

```python
def person(name, age, *, city, job):
    print(name, age, city, job)
```

- 要求：调用时必须写参数名；没有可变参数时，* 不能省略

### 2.4 递归函数

- 底层实现：函数调用依靠调用栈（stack），每调用一层增加一个栈帧，返回时销毁栈帧，栈满就溢出报错。
- 尾递归：return 直接返回函数自身调用，return 后不能有表达式。事实上尾递归和循环的效果是一样的，所以，把循环看成是一种特殊的尾递归函数也是可以的。
- 理想效果：尾递归优化后只占用 1 个栈帧，避免栈溢出。

重要：Python 解释器本身没有做尾递归优化，写了尾递归照样会栈溢出。

## 3. 高级特性

### 3.1 切片

切片语法：L[起始索引:结束索引:步长]

- 起始索引为 0 可以省略：L[:3]
- 支持负数索引（倒数）：L[-1] 代表最后一个元素，同样支持倒数切片

示例：

```python
L = list(range(100))
L[:10:2]   # 前10个数，每2个取1个
L[::5]     # 全部元素，每5个取1个
```

- 适用对象：list、tuple、字符串 str，都支持切片操作

### 3.2 迭代

迭代：用 for ... in 遍历对象，区别于 C 语言依靠下标循环。

- dict 默认迭代 key，存储无序，迭代结果顺序可能不一致

```python
d = {'a': 1, 'b': 2, 'c': 3}
for key in d:
    print(key)

for val in d.values():   # 迭代 value
    print(val)

for k, v in d.items():   # 同时迭代 key 和 value
    print(k, v)
```

- 判断是否是可迭代对象 Iterable

```python
from collections.abc import Iterable
isinstance('abc', Iterable)
isinstance([1, 2, 3], Iterable)
isinstance(123, Iterable)
```

- enumerate()：把列表转为【索引-元素】对，同时拿到下标和值

```python
for i, value in enumerate(['A', 'B', 'C']):
    print(i, value)
```

### 3.3 列表生成式

一行简洁代码快速生成 list，替代普通 for 循环。

```python
# 普通循环写法
L = []
for x in range(1, 11):
    L.append(x * x)

# 列表生成式写法
[x * x for x in range(1, 11)]

# 带 if 筛选（if 写在 for 后面，只筛选，不能写 else）
[x * x for x in range(1, 11) if x % 2 == 0]

# 双层循环，生成全排列
[m + n for m in 'ABC' for n in 'XYZ']

['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

列表生成式也可以使用两个变量来生成list：

>>> d = {'x': 'A', 'y': 'B', 'z': 'C' }

>>> [k + '=' + v for k, v in d.items()]

['y=B', 'x=A', 'z=C']
```

⚠️ if 位置区分（高频易错点）：

- for 后面的 if：用于过滤筛选，不能加 else，如 [x for x in range(1,11) if x%2==0]
- for 前面的 if：属于表达式，必须搭配 else，否则报错，如 [x if x % 2 == 0 else -x for x in range(1, 11)]

### 3.4 生成器

一边循环、一边计算，不一次性把所有元素存入内存，节省空间。

创建方式 1：列表生成式 [] 改成 ()

```python
L = [x * x for x in range(10)]   # list，全部存内存
g = (x * x for x in range(10))   # generator
```

取值方式：

- next(g)：手动取下一个元素，取完抛出 StopIteration（不推荐日常写）
- for 循环遍历（推荐，generator 也是可迭代对象）

```python
g = (x * x for x in range(10))
for n in g:
    print(n)
```

创建方式 2：函数里写 yield

包含 yield 的函数 = 生成器函数，调用返回 generator 对象。以斐波拉契数列为例：

```python
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

f = fib(6)

print(next(f))

print(next(f))

print(next(f))

print(next(f))

print(next(f))

print(next(f))

或者

for n in fib(6):

    print(n)
```

a, b = b, a + b 等价于先打包元组再解包：t = (b, a+b); a = t[0]; b = t[1]

执行特点：

- 普通函数：顺序执行，遇到 return 结束
- generator 函数：调用 next() 才运行，碰到 yield 暂停并返回；下一次 next() 从上次暂停处继续执行

注意：

- 直接 for 循环遍历 generator，拿不到 return 的返回值；想要拿到，必须捕获 StopIteration，返回值存在异常对象的 .value 中

```python
g = fib(6)
while True:
    try:
        x = next(g)
        print('g:', x)
    except StopIteration as e:
        print('Generator return value:', e.value)
        break
```

### 3.5 迭代器

- Iterable（可迭代对象）：可以直接放在 for 循环里，如 list、dict、str、generator
- Iterator（迭代器）：可以被 next() 不断调用、逐个产出数据，generator 属于 Iterator

```python
from collections.abc import Iterator
isinstance((x for x in range(10)), Iterator)  # True
isinstance([], Iterator)                     # False
isinstance({}, Iterator)                     # False
isinstance('abc', Iterator)                  # False
```

- 转换：iter() 将 Iterable 转为 Iterator

```python
isinstance(iter([]), Iterator)
isinstance(iter('abc'), Iterator)
```

核心特点：

- Iterator 是惰性数据流，只有调用 next() 才计算下一个元素，无法提前知道序列长度
- Iterator 可以表示无限大的数据流（如全体自然数），list 做不到

## 4. 函数式编程

函数式编程：抽象程度很高的编程范式。

- 纯函数：无变量，输入确定则输出一定确定，**无副作用**
- 有副作用：允许变量，函数内部状态可变，相同输入可能得到不同输出
- 特点：函数可以作为参数传入、也可以作为返回值
- Python：仅部分支持函数式编程，允许变量，**不是纯函数式语言**

### 4.1 高阶函数

高阶函数：可以接收另一个函数作为参数的函数。

原理：变量可以指向函数，函数名本身就是指向函数的变量。

```python
f = abs
f(-10)   # 等价 abs(-10)
```

最简高阶函数模板：

```python
def add(x, y, f):
    return f(x) + f(y)
```

#### （1）map/reduce

map(函数, Iterable)

- 将传入函数依次作用在序列每个元素上
- 返回 Iterator（惰性序列），一般用 list() 转列表查看结果

```python
def f(x):
    return x * x
r = map(f, [1,2,3,4,5,6,7,8,9])
list(r)
# [1, 4, 9, 16, 25, 36, 49, 64, 81]

list(map(str, [1,2,3,4,5,6,7,8,9]))
# ['1', '2', '3', '4', '5', '6', '7', '8', '9']
```

**reduce()**

需要导入：from functools import reduce

reduce(二元函数, 序列)

- 传入函数必须接收2个参数
- 累积计算：reduce(f, [x1,x2,x3,x4]) = f(f(f(x1,x2),x3),x4)

```python
from functools import reduce
def add(x,y):
    return x+y
reduce(add, [1,3,5,7,9])   # 25
```

字符串转数字示例：

```python
from functools import reduce
DIGITS = {
    '0':0,'1':1,'2':2,'3':3,'4':4,
    '5':5,'6':6,'7':7,'8':8,'9':9
}
def char2num(s):
    return DIGITS[s]

def str2int(s):
    return reduce(lambda x,y: x*10+y, map(char2num, s))
```

#### （2）filter

filter(判断函数, 序列)

- 函数返回 True / False
- 保留返回 True 的元素，丢弃 False 的元素
- 返回 Iterator

```python
# 保留奇数
def is_odd(n):
    return n % 2 == 1
list(filter(is_odd, [1,2,4,5,6,9,10,15]))
# [1, 5, 9, 15]

# 去掉空字符串
def not_empty(s):
    return s and s.strip()
list(filter(not_empty, ['A','','B',None,'C','  ']))
# ['A', 'B', 'C']
```

埃氏筛法求素数（生成器 + filter）

```python
def _odd_iter():
    n = 1
    while True:
        n += 2
        yield n

def _not_divisible(n):
    return lambda x: x % n > 0

def primes():
    yield 2
    it = _odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(_not_divisible(n), it)

# 打印100以内素数
for n in primes():
    if n < 100:
        print(n)
    else:
        break
```

#### （3）sorted

sorted(序列, key=函数, reverse=False)

- 高阶函数，key 指定排序规则
- reverse=True 开启降序

```python
sorted([36,5,-12,9,-21], key=abs)
# [5, 9, -12, -21, 36]

sorted(['bob','about','Zoo','Credit'], key=str.lower)
# ['about', 'bob', 'Credit', 'Zoo']
```

### 4.2 返回函数

高阶函数可以把函数作为返回值返回。

```python
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax += n
        return ax
    return sum

f = lazy_sum(1,3,5,7,9)
f()   # 调用时才计算，返回25
```

闭包：内部函数引用外部函数的局部变量/参数，外部函数返回内部函数，变量被绑定保存。

- 每次调用 lazy_sum() 都会生成全新函数对象，即使参数相同，f1 == f2 → False

⚠️ 闭包高频坑：返回函数不要引用循环变量或者后续会变化的变量

```python
# 错误写法，全部返回9
def count():
    fs = []
    for i in range(1,4):
        def f():
            return i*i
        fs.append(f)
    return fs

# 正确写法：用参数绑定当前循环值
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []
    for i in range(1,4):
        fs.append(f(i))
    return fs
```

nonlocal：在内层修改外层函数局部变量

```python
def inc():
    x = 0
    def fn():
        nonlocal x
        x = x + 1
        return x
    return fn
f = inc()
```

### 4.3 匿名函数

语法：lambda 参数: 表达式

- 只能有**一个表达式**，自动返回表达式结果，不用写 return
- 无函数名，避免命名冲突；本身是函数对象，可以赋值、作为返回值

```python
list(map(lambda x: x*x, [1,2,3,4,5,6,7,8,9]))

# 赋值给变量
f = lambda x: x*x
f(5) # 25

# 作为返回值
def build(x, y):
    return lambda: x*x + y*y
```

### 4.4 装饰器

装饰器：**在不修改原函数代码的前提下，运行时动态给函数增加功能**

本质：是一个**返回函数的高阶函数**

函数对象自带 `__name__` 属性，可以获取函数名称：

```python
now.__name__
```

#### （1）两层装饰器（不带参数）

语法：@装饰器名，写在函数定义上方

等价代码：now = log(now)

```python
import functools
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log
def now():
    print('2024-6-1')
```

- wrapper(`*args`, `**kw`)：适配任意参数的原函数
- 执行顺序：先跑wrapper内新增代码，再调用原始函数
- @functools.wraps(func)：修复装饰后函数 `__name__` 变成 wrapper 的问题

#### （2）三层装饰器（装饰器自身需要传参数）

等价代码：now = log('execute')(now)

1. 先执行 log('execute')，返回 decorator

2. 再执行 decorator(now)，返回 wrapper

```python
import functools
def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log('execute')
def now():
    print('2024-6-1')
```

### 4.5 偏函数

functools.partial：固定原函数的一部分参数，生成一个新函数

作用：简化频繁传固定参数的场景

```python
int('12345', base=8)
int('12345', base=16)
```

手动封装版本：

```python
def int2(x, base=2):
    return int(x, base)
```

用 partial 创建偏函数：

```python
import functools
int2 = functools.partial(int, base=2)
int2('1000000')
```

- int2 = functools.partial(int, base=2)：固定关键字参数 base=2
- 调用 int2('10010') 等价于 int('10010', base=2)

固定位置参数 `*args`：

```python
max2 = functools.partial(max, 10)
max2(5,6,7)
# 等价 max(10,5,6,7) → 结果10
```

partial 接收三类参数：函数对象、`*args`（固定位置参数）、`**kw`（固定关键字参数）

## 5. 模块

### 使用模块

模块：一个 .py 文件就是一个模块。

好处：

1. 代码分文件，可维护性高

2. 可以复用，导入内置/第三方模块

3. 隔离命名，避免变量、函数名冲突

#### （1）模块与包

包（Package）：用目录组织模块，解决模块重名冲突

- 包目录下必须有 `__init__`.py，否则只是普通文件夹
- `__init__`.py 可以为空，也可以写代码

示例目录：

```text
mycompany
├─ __init__.py
├─ abc.py
└─ xyz.py
```

abc.py 的模块名：mycompany.abc；xyz.py：mycompany.xyz

支持多级包：mycompany.web.www

#### （2）模块示例 hello.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' a test module '
__author__ = 'Michael Liao'

import sys
def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
```

- sys.argv：命令行参数列表，第一个元素永远是文件名
- if `__name__` == '`__main__`'：
  - 直接运行这个文件：`__name__` 为 `__main__`，执行内部测试代码
  - 被别的文件 import 导入：不执行里面的测试代码

#### （3）作用域约定（Python 习惯）

- public：普通名字，外部可以直接导入使用，如 abc、PI
- `__xxx__`：系统特殊变量，自带特殊功能（`__name__`、`__doc__`、`__author__`），自己尽量不要定义
- `_xxx` / _`_xxx`：约定为私有（private），**不应该**在模块外部直接调用（不是强制禁止，是编码规范）

私有函数封装示例：

```python
def _private_1(name):
    return 'Hello, %s' % name
def _private_2(name):
    return 'Hi, %s' % name

# 对外公开接口
def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```

## 6. 面向对象编程

面向对象：把程序看作一组对象集合，对象之间传递消息；Python 一切皆对象，class 用来自定义对象类型。

面向过程：dict 存数据 + 单独函数处理

```python
std1 = {'name': 'Michael', 'score': 98}
def print_score(std):
    print('%s: %s' % (std['name'], std['score']))
```

面向对象：数据和处理数据的方法封装在类里

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def print_score(self):
        print('%s: %s' % (self.name, self.score))

bart = Student('Bart Simpson', 59)
bart.print_score()
```

### 6.1 类和实例

- 类 Class：抽象模板；实例 Instance：根据类创建的具体对象。

定义类：class 类名(父类):，无合适父类写 object（所有类的根类）

```python
class Student(object):
    pass
```

创建实例：实例名 = 类名()

```python
bart = Student()
bart.name = 'Bart Simpson' # 动态绑定实例属性
```

`__init__` 构造方法：创建实例时自动执行，绑定必须的属性

第一个参数固定为 self，代表实例本身，不用手动传参

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
```

类方法：第一个参数必须是 self，调用时不用传 self

```python
bart.print_score()
```

### 6.2 访问限制

属性名以双下划线 __ 开头 → 私有变量，外部不能直接访问

```python
class Student(object):
    def __init__(self, name, score):
        self.__name = name
        self.__score = score
    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))
```

读取私有属性：get 方法

```python
def get_name(self):
    return self.__name
```

修改私有属性：set 方法，可增加参数校验

```python
def set_score(self, score):
    if 0 <= score <= 100:
        self.__score = score
    else:
        raise ValueError('bad score')
```

注意区分：

- `__xxx__`：前后双下划线，系统特殊变量，不是私有
- `_name`：单下划线，约定视为私有，语法上仍可访问

Python 私有是伪私有：底层改名 _类名__属性，如 bart.`_Student__name`

外部 bart._`_name` = "xxx" 只是新增独立属性，不会修改内部 _`_name`

### 6.3 继承和多态

继承：子类继承父类，复用父类属性与方法；子类可重写父类方法。

```python
class Animal(object):
    def run(self):
        print('Animal is running...')

class Dog(Animal):
    def run(self):
        print('Dog is running...')
    def eat(self):
        print('Eating meat...')
```

多态：同一个方法调用，运行时按对象真实类型执行对应代码

```python
def run_twice(animal):
    animal.run()
    animal.run()

run_twice(Dog())
```

开闭原则：对扩展开放（新增子类），对修改封闭（原有调用代码不用改动）

鸭子类型（Python 动态语言）：不强制继承，只要对象拥有对应方法即可使用

```python
class Timer(object):
    def run(self):
        print('Start...')
```

### 6.4 获取对象信息

type()：判断精确类型，不识别继承关系

```python
type(123) == int
type(abs)
```

isinstance()：判断是否属于该类型或其子类，支持多类型判断

```python
isinstance(h, Husky)
isinstance([1,2,3], (list, tuple))
```

dir()：列出对象全部属性、方法

```python
dir('ABC')
```

len(obj) 本质调用 obj.`__len__`()，自定义类实现 `__len__` 即可支持 len()

hasattr / getattr / setattr：动态操作对象属性

```python
hasattr(obj, 'x')
getattr(obj, 'y', 404)
setattr(obj, 'y', 19)
```

### 6.5 实例属性和类属性

- 实例属性：self.xxx，属于单个实例，各个实例相互独立

```python
class Student(object):
    def __init__(self, name):
        self.name = name

s = Student('Bob')
s.score = 90
```

- 类属性：直接写在 class 内部，归类所有，所有实例共享；实例属性优先级更高，会屏蔽同名类属性

```python
class Student(object):
    name = 'Student'

s = Student()
print(s.name)       # Student

s.name = 'Michael'  # 实例属性屏蔽类属性
print(s.name)       # Michael

del s.name          # 删除实例属性
print(s.name)       # Student，重新显示类属性
```

## 7. 面向对象高级编程

### 7.1 使用 `__slots__`

`__slots__`：限制实例可以绑定的属性，节省内存，阻止动态新增无关属性

```python
class Student(object):
    __slots__ = ('name', 'age') # 只允许 name、age

s = Student()
s.name = 'Michael'
s.age = 25
s.score = 99 # ❌ AttributeError，score 不在 slots 里
```

注意：`__slots__` 只对当前类生效，子类默认不受限制

子类如果自己定义 `__slots__`，允许的属性 = 子类自身 + 父类的 `__slots__`

补充：不写 `__slots__` 时，实例默认有 `__dict__`，可以随便绑定属性、方法

```python
from types import MethodType

def set_age(self, age):
    self.age = age

class Person(object):
    pass

p = Person()
p.set_age = MethodType(set_age, p) # 仅给单个实例绑定方法

def set_score(self, sc):
    self.score = sc

Person.set_score = set_score # 给类绑定，全部实例可用
```

### 7.2 使用 @property

@property：把方法伪装成属性访问，赋值时自动加入校验逻辑

#### （1）普通 get / set 写法

```python
class Student(object):
    def get_score(self):
        return self._score
    def set_score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

调用：s.set_score(80)、s.get_score()

#### （2）@property 写法（推荐）

```python
class Student(object):
    @property
    def score(self):
        return self._score
    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

调用：s.score = 80、print(s.score)

只写 @property、不写 @xxx.setter → 只读属性

坑：方法名不要和实例变量同名，会无限递归 RecursionError

### 7.3 多重继承 & MixIn

MixIn：多重继承用来横向添加功能，主继承描述“是什么”，MixIn 描述“能做什么”

```python
class Animal(object):
    pass
class Mammal(Animal):
    pass
class RunnableMixIn(object):
    def run(self):
        print('Running...')

class Dog(Mammal, RunnableMixIn):
    pass
```

MixIn 命名习惯以 MixIn 结尾，一般不单独实例化

一个类可以混入多个 MixIn，避免很深的继承链

Python 库例子：TCPServer + ForkingMixIn = 多进程 TCP 服务

```python
class MyTCPServer(TCPServer, ForkingMixIn):
    pass
```

### 7.4 定制类（魔法方法 `__xxx__`）

#### （1）`__str__` / `__repr__`

`__str__`：print(对象) 调用，给用户看

`__repr__`：直接敲变量显示，给调试用

```python
class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f'Student object (name={self.name})'
    __repr__ = __str__
```

#### （2）`__iter__` + `__next__`

让类支持 for ... in 迭代

```python
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1
    def __iter__(self):
        return self
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 100000:
            raise StopIteration()
        return self.a

for n in Fib():
    print(n)
```

#### （3）`__getitem__`

支持 obj[n] 下标、切片；参数可能是 int 或者 slice 对象

```python
class Fib(object):
    def __getitem__(self, n):
        if isinstance(n, int):
            # 索引取值
            a, b = 1, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        if isinstance(n, slice):
            # 切片
            start = n.start if n.start is not None else 0
            stop = n.stop
            a, b = 1, 1
            res = []
            for x in range(stop):
                if x >= start:
                    res.append(a)
                a, b = b, a + b
            return res
```

配套：`__setitem__` 赋值、`__delitem__` 删除元素

#### （4）`__getattr__`

访问不存在的属性时触发，动态返回属性 / 函数

```python
class Student(object):
    def __getattr__(self, attr):
        if attr == 'age':
            return lambda: 25
        raise AttributeError(
            f"'Student' object has no attribute '{attr}'"
        )
```

经典用途：链式 URL 构造，REST API SDK

```python
class Chain(object):
    def __init__(self, path=''):
        self._path = path
    def __getattr__(self, path):
        return Chain(f'{self._path}/{path}')
    def __str__(self):
        return self._path
    __repr__ = __str__

Chain().status.user.timeline.list # /status/user/timeline/list
```

#### （5）`__call__`

让实例可以直接像函数一样 实例() 调用

```python
class Student(object):
    def __init__(self, name):
        self.name = name
    def __call__(self):
        print(f'My name is {self.name}')

s = Student('Michael')
s()
```

callable(x) 判断对象是否可调用（函数、带 `__call__` 实例返回 True）

### 7.5 使用枚举类 Enum

```python
from enum import Enum, unique

@unique # 检查 value 不重复
class Weekday(Enum):
    Sun = 0
    Mon = 1
    Tue = 2
```

访问方式：Weekday.Mon、Weekday['Mon']、Weekday(1)

.value 获取保存的数字

遍历：for name, member in Weekday.`__members__`.items()

### 7.6 元类 metaclass

type 两个用法：

- type(obj)：查看对象类型
- type(类名, 父类元组, 属性方法字典)：运行时动态创建类

```python
def hello(self):
    print("hello")

Hello = type('Hello', (object,), dict(hello=hello))
```

元类：用来创建类的类，继承 type，重写 `__new__`，在类创建时修改类

```python
class ListMetaclass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, val: self.append(val)
        return type.__new__(cls, name, bases, attrs)

class MyList(list, metaclass=ListMetaclass):
    pass
```

经典场景：ORM 框架（元类扫描字段映射，父类提供 save）

## 8. 正则表达式 re

### 8.1 基础符号

- `\d`：数字；`\w`：字母 / 数字；`.`：任意字符；`\s`：空白（空格、tab）
- `*`：任意个（0~ 多）；`+`：至少 1 个；`?`：0 或 1 个；`{n}`：n 个；`{n,m}`：n~m 个
- `[]` 范围：`[0-9a-zA-Z_]`
- `A|B` A 或 B
- `^` 开头；`$` 结尾
- `r''` 原始字符串，不用额外转义 `\`

### 8.2 re.match

re.match(r'正则', s)：从字符串开头匹配，成功返回 Match 对象，失败 None

```python
import re
re.match(r'^\d{3}\-\d{3,8}$', '010-12345')
```

判断写法：

```python
if re.match(r'xxx', test):
    print('ok')
```

### 8.3 re.split 分割字符串

```python
re.split(r'\s+', 'a b   c') # 多个空格分割
re.split(r'[\s\,;]+', 'a,b;; c  d')
```

### 8.4 分组 ()

() 捕获分组，group(0) 整个匹配串，group(1) 第一组，groups() 返回全部分组元组

```python
m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
m.groups()
# ('010', '12345')
```

时间例子：

```python
t = '19:05:30'
m = re.match(
    r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])'
    r'\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])'
    r'\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$',
    t
)
m.groups()
```

### 8.5 贪婪匹配

默认贪婪：尽可能多匹配；加 ? 变成非贪婪（尽可能少）

```python
re.match(r'^(\d+)(0*)$', '102300').groups()
# 贪婪：('102300', '')

re.match(r'^(\d+?)(0*)$', '102300').groups()
# 非贪婪：('1023', '00')
```

### 8.6 预编译 re.compile

同一个正则反复大量使用，提前编译提升效率

```python
re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
re_telephone.match('010-12345').groups()
```

## 9. 类型注解（Type Hints）

### 9.1 类型注解基础

类型注解（Type Hints）：在代码中标注变量、函数参数、返回值的数据类型说明标签。

- 提高可读性：一眼看懂代码意图
- 便于静态检查：运行前用工具发现潜在类型错误
- 增强 IDE 支持：更准确的自动补全和提示

⚠️ 注意：注解只是标签，不是强制约束；Python 解释器运行时不校验，传错类型代码照样执行。

变量注解写法：变量名: 类型 = 值

```python
# 没有注解
name = "Alice"
age = 30

# 有注解
name: str = "Alice"        # 字符串
age: int = 30              # 整数
is_student: bool = False   # 布尔值
scores: list[int] = [95, 88, 91]  # 整数列表
```

### 9.2 函数注解

参数注解：参数名: 类型；返回值注解：-> 返回类型。

```python
# 没有注解
def greet(first_name, last_name):
    full_name = first_name + " " + last_name
    return "Hello, " + full_name

# 有注解
def greet(first_name: str, last_name: str) -> str:
    full_name = first_name + " " + last_name
    return "Hello, " + full_name
```

注解和默认值可以同时使用：

```python
def say_hello(name: str, times: int = 1) -> str:
    return " ".join([f"Hello, {name}!"] * times)

print(say_hello("Bob"))
print(say_hello("Alice", 3))
```

### 9.3 容器类型注解

Python 3.9+ 推荐原生写法：list[int]、dict[str, int]。

旧版本用 typing：List、Dict、Tuple、Set。

```python
# 原生写法（3.9+）
numbers: list[int] = [1, 2, 3, 4, 5]
student_scores: dict[str, int] = {"Alice": 95, "Bob": 88}
person_info: tuple[int, str, bool] = (25, "Alice", True)
unique_names: set[str] = {"Alice", "Bob", "Charlie"}

# typing 写法（旧版本）
from typing import List, Dict, Tuple, Set
numbers: List[int] = [1, 2, 3, 4, 5]
student_scores: Dict[str, int] = {"Alice": 95, "Bob": 88}
person_info: Tuple[int, str, bool] = (25, "Alice", True)
unique_names: Set[str] = {"Alice", "Bob", "Charlie"}
```

### 9.4 Optional 与 Union

#### （1）Optional

表示：可能是某类型，也可能是 None。

等价写法：Optional[T] = Union[T, None]。

```python
from typing import Optional

def find_student(name: str) -> Optional[str]:
    """根据名字查找学生，可能找到也可能返回 None"""
    students = {"Alice": "A001", "Bob": "B002"}
    return students.get(name)  # 可能返回字符串或 None
```

#### （2）Union

表示：可能是多种类型中的一种。

```python
from typing import Union

def process_input(data: Union[str, int, list[int]]) -> None:
    """处理可能是字符串、整数或整数列表的输入"""
    if isinstance(data, str):
        print(f"字符串: {data}")
    elif isinstance(data, int):
        print(f"整数: {data}")
    elif isinstance(data, list):
        print(f"列表: {data}")
```

### 9.5 静态类型检查 mypy

安装与运行：

```bash
pip install mypy
mypy example.py
```

检查示例（传入字符串给 int 参数）：

```python
# example.py
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers("5", "3")  # 类型错误
```

mypy 报错：

```text
example.py:4: error: Argument 1 to "add_numbers"
has incompatible type "str"; expected "int"
Found 2 errors in 1 file (checked 1 source file)
```

IDE 集成（VSCode / PyCharm）：错误高亮、智能提示、自动补全。

mypy 只做静态检查，不修改代码、不影响 Python 正常执行。

### 9.6 最佳实践与易错点

#### （1）最佳实践

- 渐进式采用：新代码优先加注解，旧代码逐步补充，不必一次全加
- 保持一致性：项目内统一注解风格，团队协商详细程度
- 优先给函数接口、公共方法加注解；第三方库可安装对应的类型存根包（如 types-xxx），或临时用 Any 绕过

#### （2）易错考点

- 类型注解只做提示，运行时不会校验，传错类型照样运行
- Optional[T] = Union[T, None]；-> None 表示函数没有返回值
- Python 3.9+ 优先用原生泛型 list[int]，替代旧的 List[int]
