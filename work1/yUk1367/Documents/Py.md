#「 基础容器 」
###List
```Python
l = [0,'cat',1.14,False] #可以创建空列表
```
```Python
list(range(0,10))
```

------------

 - 索引
 ```Python
l[0] #第一个元素
l[-1] #倒数第一个元素
```

 - 切片
```Python
l[start:stop<:step>] #不包含stop
#step为负则逆向
```

 - 运算
 ```Python
l + [(8,21)]
l * 4
```

 - 
 ```Python
'cat' in l #判断元素是否存在
```
 ```Python
len(l) #获取列表长度
```

------------

 - 修改
```Python
l[1] = 'dog'
```

 - 添加
   - 
   ```Python
l.append([-1+j]) #在末尾
```
   - 
   ```Python
l.insert(-1,[-1+j])
```

 - 删除
   - 
```Python
del l[2]
```
   - 
   ```Python
   l.remove(1.14) #删除第一个匹配项
```
   - 
```Python
   l.clear() #清空
```

------------

- 转换类型
 - 分割字符串
```Python
'I*am*so*happy*'.split('*') #分割符号默认为空格
```
   输出 `['I','am','so','happy','']`

 - 连接列表
```Python
' '.join(l)
```
输出 `'0 cat 1.14 False'`

------------

###Dictionary
```Python
d = {'Name':'Yukari','Age':17} #可以创建空字典
```

------------

 - 访问
 ```Python
d['Name'] #返回对应key的value
```

- 修改/添加
```Python
d['Age'] = 1200
```

- 删除
 - ```Python
del d['Age']
del d #删除整个字典
   ```
 - ```Python
   d.clear() #删除所有元素
   ```
   
#「 函数 」
###Lambda（匿名函数）
```Python
func = lambda <a,b,c,...> : expression
```
- 不需要`def`关键字便可以简单定义一个函数。
- 返回expression的结果。

###Decorator（装饰器）
```Python
def decorator_func(func):
	def wrapper():
		#前置操作
		result = func() #执行原函数
		#后置操作
		return result
	return wrapper
```
```Python
@decorator_func
def decorated_func():
	...
```
- 流程：
1.装饰器函数接收一个函数对象func作为参数；
2.在装饰器内部定义一个新的包装函数wrapper；
3.将包装函数返回取代原来的函数。
- 本质是一个嵌套函数。
- 外层再进行一次嵌套可以使装饰器本身接受参数。
- 多个装饰器的叠加作用顺序为从上到下。
- 装饰器也可以作用于Class。

#「 面向对象 」
###Class
```Python
class NewClass:
	def __init__(self,a,...): #初始化方法，可省略
		self.a = ... #实例变量（实例的独立属性）
		... 
	#类变量（类的共有属性）
	i = ... #公开属性
	_j = ... #受保护属性（约定）
	__k = ... #私有属性（外部无法访问）
	def method1(self,b,...):
		b= ... #局部变量（存在于方法内部）
		self.method2(...) #调用类中其他方法
	def method2(self,...):
		...

c1 = NewClass(a1,...) #实例化
c2 = NewClass(a2,...)
...

#访问类的属性与方法
c1.i
c2.method(b2)
```
- 实例方法中第一个参数 `self` 是必须的，代表类的实例（类方法第一个参数 `cls` 表示类本身，静态方法则不需要特定的第一个参数）。
- `__`也可以用在方法开头表示私有方法

------------

- 继承
```Python
class DerivedClass(BaseClass):
	...
```
子类会继承父类的属性与方法。
 - 可以继承多个父类。
 - 当方法重名时，子类的方法会覆盖父类（可以通过`super()`用子类对象调用父类已被覆盖的方法）。
 
###Magic Methods
- object类是所有类的父类，它提供了许多内置方法，即魔法方法。
- 魔法方法以`__`开头和结尾，如上述的 `__init__` 初始化方法。
- 由继承的性质，魔法方法支持重载，使得我们可以自定义对象的操作，例如用魔法方法实现自定义迭代器。
###面向对象编程（OOP）
- 类定义对象的属性（包含类中的数据成员）与方法，对象作为类的一个实例将属性与方法“封装”。
- 使代码模块化，减少不必要的代码重复（可复用），同时提高代码的可扩展性，“封装”也使数据更加安全。

#「 文本处理 」
###正则表达式
- 正则表达式是一种匹配字符串中字符组合的模式，可以用于查找、替换、验证和提取文本数据。

------------

 - 字符分为**普通字符**与**元字符**。

- 普通字符
 包括**可打印字符**与**不可打印字符**。
```
[ ]  #匹配其中任意一个字符
[^ ] #匹配除其中外任意一个字符
.    #匹配除了换行符外的任意一个字符
```
```
\w   #数字、字母、下划线
\s\S #所有（空白符与非空白符）
a-z / A-Z  #小/大写字母
0-9 / \d   #数字
\cx #Ctrl + x(x为a-z / A-Z，否则c视为原义'c')
...
```

- 特殊字符
 ```
 ?     #匹配前面零或一次
 *     #匹配前面零或多次
 +     #匹配前面一或多次
 {n}   #匹配前面n次
 {n,}  #匹配前面至少n次
 {n,m} #匹配前面n到m次
 ?     #非贪婪或最小匹配
 ```
 ```
^  #匹配字符串的开头
$  #匹配字符串的结尾
\b #匹配单词边界(单词与空格之间)
\B #匹配非单词边界
 ```
 ```
() #捕获分组，可索引多个匹配值（0为完整匹配的整个正则表达式）
|  #指定选择 “或”
?: #消除匹配缓存（包括以下带?的都是非捕获元）
exp1(?=exp2) #查找exp2前面的exp1
(?<=exp2)exp1 #查找exp2后面的exp1
exp1(?!exp2) #查找后面不是exp2的exp1
(?<!exp2)exp1 #查找前面不是exp2的exp1
\1 #反向引用缓冲区1（最多储存99个子表达式，\10，若缓冲区10无储存则指八进制字符码）
```
 - 分组也可以通过命名区分 `(?P<name>pattern)`
 反向引用 `	(?P=name)`

------------
- **修饰符**用于改变正则表达式的匹配行为，写在表达式后。
```
i #忽略大小写
g #全局匹配（非匹配第一个后停止）
m #多行模式（使^与$匹配每行的开头和结尾而非整个字符串）
s #单行模式（使.的匹配包含换行符）
u #Unicode模式
y #粘性匹配（从lastIndex指定位置开始）
x #扩展模式（忽略正则表达式中的空白和注释）
Python特有:
re.A #仅匹配ASCII字符
re.L #本地化设置
```

- Python中在参数 `flag` 使用re.N作为修饰符，并且可以使用|指定多个修饰符。
- 可以在仅括号内使用修饰符或关闭修饰符，如(?imx)与(?-imx)。
- Python中re.VERBOSE或re.X允许使用`#`添加注释，即上述扩展模式，此时匹配空白字符需要\ 或\s等转义。

------------

```Python
import re
```

- 匹配

  ```Python
re.match(pattern,string,flags=0) #匹配开头，返回匹配的对象或 None
re.search(pattern,string,flags=0) #整个字符串，返回第一个匹配的对象或 None
group(num) #返回分组内容（可返回多个，省略参数等同于num=0）
groups() #返回包含所有分组内容的元组
start()/end() #返回开始/结束位置
span() #返回包含开始和结束的元组
re.findall(pattern, string, flags=0) #匹配所有，返回一个列表，存在多个分组返回元组列表，没有则返回空列表
pattern.findall(string[, pos=0[, endpos]]) #pos：指定起始位置 endpos：结束位置，默认字符串长度
re.finditer(pattern, string, flags=0) #类似，把对象作为迭代器返回
```

- 修改

```Python
re.sub(pattern, repl, string, count=0, flags=0) #替换内容。repl：替换字符串（或函数，此时会自动接收匹配对象作为函数的参数） count：匹配后的最大次数（0为全替换），修饰符
re.split(pattern, string, maxsplit=0, flags=0) #将匹配位置分割后返回列表，有分组时会将分隔符保留。maxsplit：分割次数（0为无限制）
```

- 预编译

```Python
re.compile(pattern, flags) #生成一个正则表达式对象
```

#「 代码美学 」
###列表推导式
 ```Python
[expression for variable in list <if condition>]
```
- 可以迭代list将variable传入expression，并将返回值构建成一个新的列表。
- 可以通过if条件来筛选需要的值。
- 此外还有字典推导式、集合推导式、元组推导式，用法大体一致。本质都是由一个从序列到一个新序列的构建。


###Type Hints（类型注解）
- Python不同于如C这样的语言，需要提前声明类型，由解释器运行时自动推断，类型注解可以提高代码的可维护性。
#####简单注解
- 变量/参数注解 `Variable/Parameter: Type`
- 函数返回值注解 `def func() -> Type `
#####复杂注解
- `typing`模块
 - 容器内类型 `Variable: Container[Type,...]`
 - 可选类型 `Optional[Type]`
 - 联合类型 `Union[Type1,Type2,...]`
   - 其中 `Optional[Type]` 等价于 `Union[Type,None]`

#「 进阶技巧 」
###Generator（生成器）
- 使用了`yield`关键字的函数被称为生成器，本质上是一个迭代器。
- 当使用了yield时，函数会暂停执行并返回yield后面表达式的值；当使用`next()`或`for`循环时，函数会继续执行直至再次遇到yield。
- 优势是可以逐步产生值而非一次性全部返回，使迭代更方便。