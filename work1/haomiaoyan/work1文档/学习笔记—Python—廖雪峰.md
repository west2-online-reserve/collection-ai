---
tags:
  - python
  - West2-OnLineAI
  - 知识
date: 2025-10-18
---

网站链接[^1]—[python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)( **==本文的大部分内容都是取自于该文章==** ) 

[^1]: 该文档大概会持续更新吧（应该？如果想起来的话）
## 字符串

- 在最新的Python 3版本中，字符串是以Unicode编码的，也就是说，Python的字符串支持多语言
- 对于单个字符的编码，Python提供了 **`ord()`函数** 获取字符的整数表示，**`chr()`** 函数把编码转换为对应的字符
- ```python
>>> ord('A')
	65
>>> ord('中')
	20013
>>> chr(66)
	'B'
>>> chr(25991)
	'文'

  ```
### 格式化输出
  - 在Python中，采用的格式化方式和C语言是一致的，用`%`实现，举例如下：
  - ```python
>>> 'Hello, %s' % 'world'
'Hello, world'
>>> 'Hi, %s, you have $%d.' % ('Michael', 1000000)
'Hi, Michael, you have $1000000.'

    ```

### 列表与元组
- 列表与元组是相似的，不过区别在于元组 ==一旦定义了就不能进行修改== --这样可以保证代码的安全性


## 模式匹配(类似于switch-case)

### match语句

- 此语句可在 `if-elif-else`  过长时考虑

- ```python
    score = 'B'
if score == 'A':
    print('score is A.')
elif score == 'B':
    print('score is B.')
elif score == 'C':
    print('score is C.')
else:
    print('invalid score.')

  ```
  
  - 可以看到 `if-elif-else` 过于冗余，所以我们可以用 `match` 语句
  
  - ```python
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

    ```

- 还可以进行复杂匹配
- ```python
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

  ```
  - 可以匹配多个值、匹配一定范围，并且把匹配后的值绑定到变量

### 匹配列表
- 直接来看代码
- ```python
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

  ```
- 第一个`case ['gcc']`表示列表仅有`'gcc'`一个字符串，没有指定文件名，报错；

- 第二个`case ['gcc', file1, *files]`表示列表第一个字符串是`'gcc'`，第二个字符串绑定到变量`file1`，后面的任意个字符串绑定到`*files`（符号`*`的作用将在[函数的参数](https://liaoxuefeng.com/books/python/function/parameter/index.html)中讲解），它实际上表示至少指定一个文件；

- 第三个`case ['clean']`表示列表仅有`'clean'`一个字符串；

- 最后一个`case _`表示其他所有情况。

##  循环

### for循环 
- `for x in ...`循环就是把每个元素代入变量`x`，然后执行缩进块的语句。
- 而再来还需要顺便介绍一个函数—— ==`range`==  
- `range(101)`就可以生成0-100的整数序列，计算如下：
- ```python
  sum = 0
for x in range(101):
    sum = sum + x
print(sum)
  ```
  

### while 循环

#### contine语句
- 如果我们想只打印奇数，可以用`continue`语句跳过某些循环：
- ```python
  n=0
  while n < 10:
	  n = n + 1
	  if n % 2 == 0:
		  contine  #contine语句会直接开启下一轮的循环，所以在他之后的语句就不会进行
	print(n)
  ```

## set结构
- set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，**没有重复的key** 

- 要创建一个set，用`{x,y,z,...}`列出每个元素：

```plain
>>> s = {1, 2, 3}
>>> s
{1, 2, 3}
```

- 或者提供一个list作为输入集合：

```plain
>>> s = set([1, 2, 3])
>>> s
{1, 2, 3}
```

- 注意，传入的参数`[1, 2, 3]`是一个list，而显示的`{1, 2, 3}`只是告诉你这个set内部有1，2，3这3个元素，显示的顺序也不表示set是有序的

- ==重复元素在set中自动被过滤== ：

```plain
>>> s = {1, 1, 2, 2, 3, 3}
>>> s
{1, 2, 3}
```

- set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：

```plain
>>> s1 = {1, 2, 3}
>>> s2 = {2, 3, 4}
>>> s1 & s2
{2, 3}
>>> s1 | s2
{1, 2, 3, 4}
```

## 函数

### 函数的参数

- 默认参数就排上用场了。由于我们经常计算x2，所以，完全可以把第二个参数n的默认值设定为2：
 ```python
def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
```

- 这样，当我们调用`power(5)`时，相当于调用`power(5, 2)`：

```plain
>>> power(5)
25
>>> power(5, 2)
25
```

#### 关键字参数
- 关键字参数是 Python 函数中一种特殊的参数形式，它允许你在调用函数时传入**0 个或任意多个带有参数名的参数**，这些参数在函数内部会被自动组装成一个字典（`dict`），其中字典的键是参数名，值是对应的参数值
- 下面通过几个具体例子，进一步理解关键字参数的用法和作用：

##### 例 1：基础用法（传入多个关键字参数）

定义一个函数，接收必选参数和关键字参数，用于打印用户信息：

```python
def user_info(username, age, **kw):
    print(f"用户名：{username}，年龄：{age}")
    print("额外信息：", kw)  # kw是组装后的字典

# 调用时传入2个关键字参数
user_info("小明", 18, 性别="男", 爱好="篮球", 城市="北京")
```

```plaintext
用户名：小明，年龄：18
额外信息： {'性别': '男', '爱好': '篮球', '城市': '北京'}
```

这里`性别="男"`、`爱好="篮球"`等都是关键字参数，它们被自动组装成字典`{'性别': '男', '爱好': '篮球', '城市': '北京'}`，通过`kw`在函数内部使用。

##### 例 2：传入 0 个关键字参数

如果不需要额外信息，关键字参数可以省略：

```python
# 只传必选参数，不传关键字参数
user_info("小红", 20)
```

```plaintext
用户名：小红，年龄：20
额外信息： {}  # 空字典
```

此时`kw`是一个空字典，函数依然能正常执行，体现了灵活性。

##### 例 3：结合位置参数，关键字参数顺序不影响

关键字参数因为带有参数名，所以传入顺序不影响结果：

```python
# 交换关键字参数的顺序
user_info("小李", 22, 城市="上海", 性别="女")
```

```plaintext
用户名：小李，年龄：22
额外信息： {'城市': '上海', '性别': '女'}
```

虽然`城市`和`性别`的顺序换了，但`kw`字典的内容只和参数名与值有关，和传入顺序无关。

##### 例 4：在函数内部处理关键字参数（条件判断）

可以在函数内部通过字典的方式（如`kw.get()`）处理关键字参数，提取需要的信息：

```python
def product(name, price, **kw):
    print(f"商品：{name}，单价：{price}元")
    # 从关键字参数中提取“折扣”和“库存”（如果有的话）
    discount = kw.get("折扣", 1)  # 默认折扣为1（无折扣）
    stock = kw.get("库存", 0)     # 默认库存为0
    print(f"折后价：{price * discount}元，库存：{stock}件")

# 传入“折扣”关键字参数
product("笔记本", 5000, 折扣=0.9, 颜色="银色")
# 传入“库存”关键字参数
product("鼠标", 100, 库存=200)
# 不传入额外关键字参数
product("键盘", 200)
```


```plaintext
商品：笔记本，单价：5000元
折后价：4500.0元，库存：0件
商品：鼠标，单价：100元
折后价：100元，库存：200件
商品：键盘，单价：200元
折后价：200元，库存：0件
```

这里通过`kw.get()`安全地获取关键字参数（即使参数不存在也不会报错，会返回默认值），体现了关键字参数在处理 “可选额外信息” 时的实用性

#### 命名关键字参数

这里的核心是理解**命名关键字参数的两个关键特性**：**参数名称被限制**（只能传指定的名字）和**必须用关键字形式传入**（不能像位置参数那样按顺序传值），而`*`就是用来标记 “从这里开始是命名关键字参数” 的分隔符。

##### 拆解说明：

###### 1. 先看函数定义的结构

```python
def person(name, age, *, city, job):
    print(name, age, city, job)
```

- `name`、`age`：是**必选位置参数**（必须传值，且可以按位置顺序传入，比如`person('Bob', 25, ...)`）。
- `*`：**特殊分隔符**，它的作用是 “画一条线”—— 告诉 Python：`*`后面的参数（`city`、`job`）不是普通的位置参数，而是**命名关键字参数**。

##### 2. 命名关键字参数的核心规则（由`*`分隔后生效）：

- **必须用关键字形式传入**：调用时，`city`和`job`不能像`name`、`age`那样按位置顺序传值，必须写成`city=值`、`job=值`的形式。
    
    ✅ 正确调用：`person('Bob', 25, city='Beijing', job='Engineer')`（`city`和`job`带参数名）
    
    ❌ 错误调用：`person('Bob', 25, 'Beijing', 'Engineer')`（试图按位置传`city`和`job`，会报错）
    
- **参数名称被严格限制**：只能传入`*`后面指定的参数名（这里是`city`和`job`），不能传其他名字的关键字参数。
    
    ❌ 错误调用：`person('Bob', 25, city='Beijing', salary=10000)`（`salary`不是命名关键字参数，会报错）
    

##### 3. 为什么需要`*`这个分隔符？

如果没有`*`，函数定义会变成`def person(name, age, city, job):`，此时`city`和`job`是**普通的位置参数**（可以按位置传值，也能按关键字传值，且不限制额外传入其他关键字参数）。

而`*`的存在，就是为了**明确区分 “位置参数” 和 “命名关键字参数”**，强制后者必须用关键字形式传入，同时限制只能传指定的名称，避免调用时因参数顺序错误或传入无关参数导致的问题。

##### 对比：命名关键字参数 vs 普通关键字参数（`**kw`）

- 普通关键字参数`**kw`：可以接收**任意名字**的关键字参数（比如`city`、`job`、`salary`等都行），灵活性高但不限制范围。
- 命名关键字参数（如`*, city, job`）：只能接收**指定名字**的关键字参数（仅`city`和`job`），且必须用关键字形式传入，规范性更强。

##### 总结

`*`在命名关键字参数中的作用是 **“分隔标记”**，它后面的参数被明确标记为 “必须用关键字形式传入，且只能是指定名称” 的参数。这样做的好处是：让函数调用更清晰（参数含义通过名称体现），同时避免传入无关参数，减少错误



如果要限制关键字参数的名字，就可以用命名关键字参数，例如，只接收`city`和`job`作为关键字参数。这种方式定义的函数如下：

```python
def person(name, age, *, city, job):
    print(name, age, city, job)
```

和关键字参数`**kw`不同，命名关键字参数需要一个特殊分隔符`*`，`*`后面的参数被视为命名关键字参数。

如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符`*`了：

```python
def person(name, age, *args, city, job):
    print(name, age, args, city, job)
```

命名关键字参数必须传入参数名，这和位置参数不同。如果没有传入参数名，调用将报错：

```plain
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

```plain
>>> person('Jack', 24, job='Engineer')
Jack 24 Beijing Engineer
```

#### 可变参数

可变参数 `*args` 是 Python 中用于处理**不确定数量的位置参数**的特殊语法，它的核心作用是：**允许函数在调用时接收任意个（包括 0 个）不带参数名的位置参数，这些参数会在函数内部自动被组装成一个元组（tuple）**，方便函数统一处理。

##### 具体作用拆解：

1. **接收 “任意数量” 的位置参数**
    
    函数定义时，我们无法预知调用者会传入多少个位置参数（比如计算多个数的和，可能传 2 个数，也可能传 10 个数）。`*args` 可以 “兜底” 接收所有多余的位置参数，避免因参数数量不匹配导致报错。

   ```python
    def sum_all(*args):
        total = 0
        for n in args:  # args是元组，可遍历
            total += n
        return total
    
    # 调用时可传任意个位置参数
    print(sum_all(1, 2))  # 传2个 → 结果3
    print(sum_all(1, 2, 3, 4))  # 传4个 → 结果10
    print(sum_all())  # 传0个 → 结果0（args是空元组()）
    ```
    
2. **自动组装为元组（tuple）**
    
    调用时传入的所有位置参数，会被 `*args` 收集并打包成一个元组。例如调用 `sum_all(1, 2, 3)` 时，函数内部的 `args` 就是 `(1, 2, 3)`，可以像操作元组一样遍历、索引这些参数。
    
3. **与其他参数配合：接收 “多余的位置参数”**
    
    当函数同时有必选参数、默认参数时，`*args` 会接收**在必选参数和默认参数之后的所有位置参数**（即 “多余的位置参数”）。
    
    示例（结合必选参数和默认参数）：
    ```python
    def func(a, b, c=0, *args):  # 顺序：必选参数a、b → 默认参数c → 可变参数*args
        print(f"a={a}, b={b}, c={c}, 多余的位置参数：{args}")
    
    func(1, 2)  # 只传必选参数 → args=()
    func(1, 2, 3, 4, 5)  # 传了a=1, b=2, c=3，多余的4、5被args收集 → args=(4,5)
    ```


