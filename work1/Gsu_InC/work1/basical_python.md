# <center> Gsu学python基础日志 </center>

## 基础容器:List,Dict的使用技巧
### list用法
- list 特点(区别于cpp中任何一种容器)：啥都能存
- list 初始化 l = []
- Type Hint 提示list内数据类型（非强制）eg:```list[int]```
#### 增
```
l = []
l.append(1024) # 1024 可以替换为任意你想要插入l列表尾部的东西 实现过程：扩容(如果需要)+加入元素到列表尾部 时间复杂度o(1)
```
#### 删
1. del
del l[index] # 删除index位的元素
del l[start:end] # 不含end
2. pop()
remove = l.pop() # 删掉l的最后一个元素 remove为最后一个元素
remove = l.pop(index) #删掉index位置的元素 
3. remove()
l.remove(value) #删除第一个值为value的元素 如果没找到返回ValueError
安全删除：
```
def safe_remove(lst,value):
    if value in lst:
        lst.remove(value)
        return True
    return False

```
#### 改
l[index] = value # 直接改
#### 查
- 访问元素
1. 索引
```
l[index] # 越界报错 
l[start:stop:step] # 取不到stop位 返回子列表 区别于 str.substr(satrt,size) step为负数可以反过来读取
```
- 报错解决方法：try
```
try:
    pass #这里写你的调用，如print(l[index])
except Indexerror:
    pass #这里写你的解决方案
```
2. 遍历
    ##### for 遍历
```
for i in list:
    pass # 这里写你的代码,注意：这种写法无法读取到索引

# 可以使用enumerate拿到index
for index,value in enumerate(l):
    pass # 此时索引为index，值为value(index,value为python中特殊写法，这两个变量名可以自取)
    #可以enumerate(l,start=num)指定开头为num 
# 可以用经典的索引访问法
length = len(l)
for i in range(0,length,1):
    pass
```
##### 迭代器遍历   

```
# 使用iter()和next()
l_iter = iter(l)
next(l_iter) #或者l_iter.__next__()，哪个顺手用哪个
可以用while,try来实现遍历
```
##### 列表推导式(一个~~很拽~~很欠揍的写法)
[你的代码 for 变量 in 列表 if 你的条件(可以不写条件)]
```
squares = [x**2 for x in range(10)]
even_squares=[x**2 for x in range(10) if x%2 == 0] #可以带条件
```
- 查元素个数
len()
```
count = len(l)
```
#### list 分割和连接
1. split() 默认按照空白字符(换行符，空格，制表符)分割 分割字符串为列表 
string.split('.')指定用.分割 列表中不会有.，可以理解为.直接被切开
2. join() 
```
'.'.join(l) # 把列表用.连起来
''.join(l) # 把列表无缝衔接
#'' 中可以是任意字符
```

### Dict用法
Dict 特点：形式有点像cpp中的map和unordered_map 有序
{key1:value1,key2:value2,......}
实现方式:哈希表(开放地址法)
初始化 empty_dict = dict() 或 empty_dict = {}

#### 增
score[new_key]  = value
#### 删
del score[key] <mark>注意</mark>：key 要在字典中 
#### 改
直接赋值
score[key] = new_value
#### 查
score[key] 如果key不存在则报错
score.get(key) 如果key不存在则返回None
## 函数:lambda函数、Decorator装饰器
### lambda函数
可以理解为没名字且只有一行无名函数
 ```
# 普通函数
def add(x,y):
    return x + y
# lambda
add_l = almbda x,y:x + y
 ```
lambda 参数列表: 返回值 # 不能print
### Decorator装饰器
模板
```
def wrapper(fn):
    def inner(*ags,**kwags): # 打包,传入参数 
        # 目标函数前做啥
        res = fn(*ags,**kwags)  # *ags,*kwags解包 res接受fn()的返回值 
        # 目标函数后做啥
        return res
    return inner
```
使用
```
@wrapper
fn
```
## 面向对象:Class类与Magic Methods,以及OPP(面向对象编程)思想
### class 类
```
class vehicle:
    def __init__(self,speed):    # __init__ 为内置方法 用于初始化实例 python内置方法 self为内置方法
        self.speed = speed  # 给对象的speed赋值
        # 可以设默认值 eg: self.default = 0
    def time(self,s):
        print("t =" s / speed)
class bike(vehicle):        # 类的继承 bike 继承于 vehicle
    pass
class car(vehicle):
    def __init__(self,speed,fuel) # 子类重新初始化，覆盖vehicle的初始化
# 定义一个变量为类
b = bike(30)
c = car(100,15)
# 调用 例如走100米
b.time(100)
```
实例方法 如上面的def speed 第一个参数为self 可调用实例属性和类属性
类方法 使用@classmethod 可修改和访问类属性 <mark>不能访问实例属性</mark> 使用cls而不用类的名字
```
class BankAccount:
    interest_rate = 0.03
    total_accounts = 0
    @classmethod
    def increment_accounts(cls): # 这样定义了一个类方法
        cls.total_account += 1 # 类数据：账号总数+1

    @classmethod
    def set_interest_rate(cls,new_rate):
        cls.interest_rate = new_rate # 重新设置利率
        print(f"利率已重设为{new_rate}")
```
静态方法 使用@staticmethod 没有self或cls函数 不能访问实例属性和类属性 只是一个普通函数，逻辑上属于这个类
```
class example
    @staticmethod
    def add(x,y):
        return a+b
print(example.add(2,3)) # 输出5
```

### Magic Methods 扩展类的特殊行为
Magic Methods 全都是python内置方法 隐式调用
eg : __init__ 就是一种魔法方法
详见表格附件
## 文本处理:re正则表达式
之前学的时候印象最深的是```.*?```表示两个东西之间最短的任意东西,然后```.*``` 表示最长的任意东西
python中的话：
```
import re # 先导入，非常简洁
# finditer 返回迭代器，拿取内容要用.group() (常用)
it = re.finditer(r"\d+","我的电话号码是10086,你的电话是10010")
for i in it:
    print(i.group())

# search 全文匹配，返回match对象，拿数据要.group,检索到一个结果直接返回
s = re.search(r"\d+","我的电话号码是10086,你的电话号码是10010")
print(s.group()) # 拿到 10086

# match 从头开始匹配
s = re.match(r"\d+","我的电话号码是10086,")
print(s.group) # 相当于 s = re.search(r"^\d+","我的电话号码是10086,")

# 预加载正则表达式
obj = re.compile(r"(\d+)")# 预加载括号中正则表达式到obj中
ret = obj.finditer("我的电话号码是10086,你的电话号码是10010")
for i in ret:
    print(i.group())
```
格式详见附件re
ps：这东西也能写爬虫(bushi)
## 代码格式:列表推导式、Type Hints(类型注释)
### 列表推导式(一个~~很拽~~很欠揍的写法,但是在创建新列表时会更高效)
[你的代码 for 变量 in 列表 if 你的条件(可以不写条件)]
### Type Hints
这东西不是强制的，而是一种注释 py3.5+才有
能提升代码质量，但是会增加开发时间
py3.5
```
from typing import List,Dict,Tuple,Set,Optional # 要先从typing导入
l = List[str] # 类似于这样
```
py3.9+
```
# 不用从typing导入基础类型了( 你想用老方法也行)
l = list[str]
```
#### 一个带类型提示的class
```
class Product:
def __init__(self,name:str,price:int,in_stock:bool):
    self.name = name
    self.price = price
    self.in_stock = in_stock
```
### Optional
Union[T,None]的简写
在可能为None的地方写Optional
可能为... ,也可能为None
格式：def(参数名:Optional = None) -> 返回值类型
eg:
```
from typing import Optional
def fn(name:Optional[str]) -> str:
    return name if name else "匿名用户"
```
```
from typing import Optional
def find_name(id:int) -> Optional[str]：
    if id == 1:
        return "666@gmail.com"
    return None
```
可选参数写法
```
def send_message(to:str,
                content:str,    # cc 是可选的
                cc:Optional[str] = None) -> bool:
```
### Union
可以是这个，也可以是那个
Union[str,int] # 可以是string，也可以是integer
Union中可以加很多 eg：Union[str,int,float] 等等
Optional[str] 相当于 Union[str,None]
Union 和 Optional 用法类似
## generator生成器
处理大数据集的好工具
只能遍历一次
使用yield返回语句，通过next执行到下一个yield的位置 或者用for遍历
```
def fibonacci_generator(n):
    a = 0
    b = 1
    count = 0
    while n < count:
        yield a
        a,b = b,a+b
        count += 1
fib_g = fibonacci_generator(5)
for i in fib_g:
    print(i) # 这样就逐一生成斐波那契数列前5个
```