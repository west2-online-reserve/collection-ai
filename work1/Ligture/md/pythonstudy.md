# 基础容器
## 1.List
列表是一个可以存储任意类型的可变容器
```python
#增添
lst = [1, 2, 3]
lst.append(4)          # 在末尾添加元素4
lst.insert(1, 1.5)     # 在索引1处插入
#删除
lst.remove(2)          # 删除元素2
lst.pop()      # 删除最后一个元素

#切片
start = 0
end = 3
step = 1
sb_lst = lst[start:end:step]
```

### 技巧:.enumerate函数同时获取索引和值
```python
# 获取列表元素及其索引
lst = ['a', 'b', 'c']
for index, value in enumerate(lst):
    print(index, value)
```
## 2.Dict
字典Dict也是一个可变容器，以键值对的形式存储数据，使用{}表示
```python
#增改
d = {'a': 1, 'b': 2}
d['c'] = 3          # 添加键值对
d['a'] = 10         # 修改键'a'的值
#删除
del d['b']         # 删除键'b'
d.pop('c')  # 删除键'c'
#读取
value = d.get('a')  # 获取键'a'的值
keys = d.keys()     # 获取所有键
values = d.values() # 获取所有值
items = d.items()   # 获取所有键值对
```
### 技巧:1.字典推导式
```python
# 创建一个字典，键为0-4，值为键的平方
d = {i: i**2 for i in range(5)}
```
### 技巧:2.合并字典
```python
# 合并两个字典
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
d1.update(d2) #d2并入d1
new_dict1 = {**d1, **d2} #python 3.5+
new_dict2 = d1 | d2 #python 3.9+
```
---

# 函数

## 1.lambda匿名函数
lambda是一种的函数 基本语法: lambda 参数: 表达式

常用于sorted() list.sort()等的参数
```python
# 按照元组的第二个元素排序
lst = [(1, 3), (2, 1), (3, 2)]
sorted_lst = sorted(lst, key=lambda x: x[1])
```

# Decorator装饰器
Decorator是一个函数，接受另一个函数作为参数，并返回一个新的函数。 常用于日志记录等。
```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"log:xxx")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    print(a+b)
add(2,3) 
'''
输出: 
log:xxx
5
'''
```

---

# 面向对象
## 1.Class和Magic Methods
class用于定义一类具有相似属性和行为的对象,如人具有年龄性别等属性,可以定义走路等行为.
magic methods是以__开头和结尾的方法,用于实现特殊行为.如__init__在初始化时调用,__str__可以定义class用字符串怎么表示.

## 2.OOP思想
面向对象编程(OOP)是一种编程思想,将复杂的问题分解成对象,通过对象的交互解决问题.相比顺序的执行代码,更加符合人类直觉,更易于管理.

---

# re正则表达式
re是python中用于对正则表达式相关操作处理的模块
正则表达式用于根据特定规则匹配文本

正则表达式常用元字符: 

. 匹配除换行符之外的任何单个字符。

\* 匹配前面的子表达式零次或多次。

\+ 匹配前面的子表达式一次或多次。

? 匹配前面的子表达式零次或一次。

^ 匹配输入字符串的开始位置。

$ 匹配输入字符串的结尾位置。

如匹配字符串是否是邮箱:.+@.+\..+ 表示匹配一个或多个任意字符,后跟@符号,再后跟一个或多个任意字符,最后是一个点和一个或多个任意字符.

```python
import re
pattern = r'.+@.+\..+'
email = "aadads@dada.com"
if re.match(pattern, email):
    print("正确")
else:
    print("错误")
```

---

## 列表推导式
列表推导式可以方便的创建列表,语法为: [表达式 for 变量 in 可迭代对象 if 条件]

如:[x ** x for x in range(10)] 表示创建一个列表,包含0到9的每个数的乘方. 
元组推导式和字典推导式类似,分别使用()和{}表示

## Type Hint类型注释
类型注释用于标注参数或函数返回值等的类型,提高可读性
```python
im_int:int = 1
im_list:list[int] = [1,2,3]
def add(a: int, b: int) -> int:
    return a + b
```

## generator生成器
在函数内部使用yield关键字的函数就是生成器,当调用生成器时,遇到yield会暂停函数并返回yield变量的值,而使用next()函数会从当前的状态继续执行直到下一个yield.
```python
def generator_onetoten():
    for i in range(1, 11):
        yield i
generator = generator_onetoten()
print(next(generator)) # 输出1
print(next(generator)) # 输出2
```