#  <center>Python笔记</center>
## 一.基础容器：List（列表）、Dict（字典）的使用技巧
### 1.列表
#### 简介
列表是有序、可变的元素集合（有序的集装箱）
#### 使用技巧
##### 1.基础
删除元素​：del list[index]：删除指定索引的元素
添加元素​：list.append(item)：在末尾添加一个元素
##### 2.列表推导式 VS 传统写法
传统：用for循环逐个 append
```
list=[]
for i in range(1,100):
   if num % 2 ==0:
      list.append(num**2)
```
列表推导式(明了)
```
list=[num** 2 for num in range(1, 100) if num % 2 == 0]
```
##### 3.切片（取子列表，做倒序、步长截取、批量修改···）
语法：list[start: end :step]（start 默认 0，end 默认 len (list)，step 默认 1）
口诀：​含头不含尾
```
list = [1,2,3,4,5,6,7]
print(list[::-1])  # 输出：[7,6,5,4,3,2,1]  (倒序)
print(list[::2])  # 输出：[1,3,5,7]
# 批量修改子列表（替换索引1-4的元素）
list[1:5] = [10,20,30]  # 注意：替换的元素个数可以和原长度不同
print(list)  # 输出：[1,10,20,30,6,7]
```
##### 4.排序
1.sorted(list)：返回一个新的排序后的列表，原列表不变。
2.list.sort()：直接修改原列表，不返回新列表。参数 reverse=True可降序排序。
```
list= [3, 1, 4, 2]
cg-list = sorted(list) # cg-list是 [1, 2, 3, 4], list 还是 [3, 1, 4, 2]
cg-list =sorted(list, reverse=True) # 倒序
list.sort() # list 变成了 [1, 2, 3, 4]
```  

### 2.字典
#### 简介
字典是键值对集合，适合通过「键」快速查找数据  
#### 使用技巧
##### 1.访问
1.dict[key]  (不推荐，会报错)
2.dict.get(key) 
##### 2.字典推导式
类似列表推导式，适合从其他容器（如列表、元组）生成字典
```
name = ['Alice', 'Bob']
score = [90, 85]
dict = {name: score for name, score in zip(name, score)}
print(dict)  # 输出：{'Alice': 90, 'Bob': 85}
```
##### 3.遍历字典  
```
dict = {'name': 'Bob', 'age': 20, 'major': 'CS'}

# 遍历键（常用）
for key in dict:  
    print(key)

# 遍历值
for value in dict.values():
    print(value)

# 遍历键值对（最常用）
for key, value in dict.items():
    print(key, value)  

```
##### 4.合并字典
1.update()（原地合并，覆盖重复键） 
2.**解包（生成新字典，不修改原字典）
```
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}

# 方法1：原地合并（dict1被修改）
dict1.update(dict2)
print(dict1)  # 输出：{'a':1, 'b':3, 'c':4}（b被覆盖）

# 方法2：** 解包（生成新字典，原字典不变）
dict3 = {**dict1,** dict2}  # 注意：后一个字典的键会覆盖前一个
print(dict3)  # 输出：{'a':1, 'b':3, 'c':4}
```

## 二.函数：Lambda匿名函数、Decorator装饰器  
### 1.Lambda匿名函数
#### 简介
Lambda 是无名称的小型函数，仅能写单个表达式，适合逻辑简单、临时使用的场景  
#### 基本语法
```
lambda 参数1, 参数2, ... : 表达式
```
**冒号后面只能有一个表达式**
#### 使用场景
##### 1.sorted() 自定义排序规则
```
# 对字典列表按“年龄”排序
students = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 17}]
sorted_students = sorted(students, key=lambda x: x['age'])  # key 接收 Lambda 函数
print(sorted_students)  # 输出：[{'name': 'Bob', 'age': 17}, {'name': 'Alice', 'age': 18}]
```
##### 2.map() 批量处理序列（映射转换）
```
# 将列表中所有数字翻倍
nums = [1,2,3]
doubled = list(map(lambda x: x*2, nums))  # map 接收 Lambda 和序列
print(doubled)  # 输出：[2,4,6]
```
##### 3.filter() 筛选数据(过滤)
```
# 筛选列表中大于10的数字
nums = [5,12,8,15]
filtered = list(filter(lambda x: x>10, nums))  # filter 接收 Lambda（条件）和序列
print(filtered)  # 输出：[12,15]
```
### 2.Decorator装饰器
#### 简介
装饰器作用是增强函数功能,即在不修改原函数代码的情况下，为函数添加新功能  
#### 核心逻辑
1.函数可以像变量一样被传递、赋值、作为参数和返回值
2.装饰器依赖闭包实现（函数嵌套 + 内部函数引用外部变量）
```
def outer(func):  # 接收原函数作为参数
    def inner():  # 内部函数：包装原函数，添加新功能
        print("函数执行前做的事")
        func()  # 调用原函数
        print("函数执行后做的事")
    return inner  # 返回内部函数（增强后的函数）

# 原函数
def hello():
    print("Hello Python!")

# 用闭包增强原函数
up_hello = outer(hello)
up_hello()  # 调用增强后的函数
```
输出
```
函数执行前做的事
Hello Python!
函数执行后做的事
```
#### 使用技巧
##### 1. 装饰器语法糖：@ 符号（简化使用）
用 @装饰器名 直接放在原函数上方，替代手动赋值，是实际开发的标准用法：
```
def outer(func):
    def inner():
        print("执行前：记录日志")
        func()
        print("执行后：日志结束")
    return inner

@outer  # 等价于 hello = outer(hello)，自动增强函数
def hello():
    print("Hello Decorator!")

hello()  # 直接调用原函数名，实际执行增强后的逻辑
```
##### 2.通用装饰器：处理带参数的函数
装饰器需用 *args（位置参数）、**kwargs（关键字参数）兼容
(*args 和 **kwargs：* / ** 有特殊含义，args/kwargs 可替换)
*：用于 “收集所有未匹配的位置参数”，将其打包成一个元组
**：用于 “收集所有未匹配的关键字参数”，将其打包成一个字典
```
def logger(func):
    def inner(*args, **kwargs):  # 接收原函数的所有参数
        print(f"调用函数：{func.__name__}，参数：{args}, {kwargs}")
        result = func(*args, **kwargs)  # 传递参数给原函数
        print(f"函数返回值：{result}")
        return result  # 返回原函数结果
    return inner

@logger
def add(a, b):
    return a + b

add(2, b=3)  # 输出：调用函数：add，参数：(2), {'b':3}  函数返回值：5  5
```
##### 3.带参数的装饰器
如果装饰器本身也需要参数，需要再嵌套一层
```
def repeat(n):  # 这一层接受装饰器的参数
    def decorator(func):  # 这一层接受被装饰的函数
        def wrapper(*args, **kwargs):
            for i in range(n):
                print(f"第{i+1}次执行:")
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)  # 装饰器带参数
def say_hello():
    print("Hello!")

say_hello()
```
输出
```
第1次执行:
Hello!
第2次执行:
Hello!
第3次执行:
Hello!
```
##### 4.多个装饰器叠加（按顺序增强）
装饰顺序：从内到外 （靠近函数的装饰器是内层，外层装饰器包裹内层）；
执行顺序：从外到内 (先执行上方装饰器的 “前置逻辑”，后执行下方装饰器的 “前置逻辑”)
```
def decorator1(func):
    def inner():
        print("装饰器1：执行前")
        func()
        print("装饰器1：执行后")
    return inner

def decorator2(func):
    def inner():
        print("装饰器2：执行前")
        func()
        print("装饰器2：执行后")
    return inner

@decorator1  # 第一个执行（外层）
@decorator2  # 第二个执行（内层）
def hello():
    print("核心函数：Hello!")

hello()
```
输出
```
装饰器1：执行前
装饰器2：执行前
核心函数：Hello!
装饰器2：执行后
装饰器1：执行后
```
解析：
1.执行decorator1的inner的前置逻辑：print("装饰器1：执行前")
2.调用decorator1的inner中的func → 即decorator2的inner
3.执行decorator2的inner的前置逻辑：print("装饰器2：执行前")
4.调用decorator2的inner中的func → 即原 hello 函数，执行：print("核心函数：Hello!")
5.执行decorator2的inner的后置逻辑：print("装饰器2：执行后")
6.回到decorator1的inner，执行其后置逻辑：print("装饰器1：执行后")
 ## 三.面向对象：Class（类）与Magic Methods(魔法方法)，以及OOP(面向对象编程)的思想
 ### 1.OOP（面向对象编程）思想  
#### 简介
OOP 的核心是将数据（属性）和操作数据的行为（方法）封装在一起
##### 特性
1.封装：把数据和方法打包在一起，隐藏内部实现细节
2.继承：子类可以继承父类的属性和方法，实现代码复用，同时可扩展新功能
3.多态：不同对象对同一消息做出不同响应
### 2.Class（类）
#### 1.基础
定义类、实例化对象
```
# 定义类（首字母大写，规范命名）
class Student:
    # 初始化方法：创建对象时自动调用，给对象绑定属性
    def __init__(self, name, age, scores):
        self.name = name  # self 指代当前对象，绑定属性（实例属性）
        self.age = age
        self.scores = scores  # 复用列表知识，存储多门成绩

    # 类的方法：操作对象属性（必须带 self 参数）
    def get_avg_score(self):
        """计算平均分（复用列表推导式/内置函数）"""
        return sum(self.scores) / len(self.scores)

    def show_info(self):
        """显示学生信息（复用字典/字符串格式化）"""
        info = {
            "姓名": self.name,
            "年龄": self.age,
            "平均分": f"{self.get_avg_score():.1f}"
        }
        for k, v in info.items():
            print(f"{k}: {v}")

# 实例化对象：类名() 调用 __init__ 方法
stu1 = Student("Alice", 18, [90, 85, 95])
stu2 = Student("Bob", 19, [80, 75, 88])

# 调用对象的方法、访问属性
stu1.show_info()
print(f"{stu1.name} 的平均分：{stu1.get_avg_score()}")
```
#### 2.继承
子类复用父类功能，扩展新属性
```
class CollegeStudent(Student):  # 括号内写父类名，实现继承
    def __init__(self, name, age, scores, major):
        super().__init__(name, age, scores)  # 调用父类初始化方法
        self.major = major  # 新增子类属性

    # 重写父类方法（多态体现）
    def show_info(self):
        super().show_info()  # 调用父类方法
        print(f"专业：{self.major}")

# 子类对象实例化
col_stu = CollegeStudent("Charlie", 20, [92, 88, 90], "计算机科学")
col_stu.show_info()  # 执行子类重写后的方法
```
#### 3.封装与访问控制
Python中没有严格的私有属性，但通过命名约定实现：
1.name：公有属性
2._name：受保护属性（提示不要直接访问）
3.__name：私有属性（会进行名称修饰）
### 3.Magic Methods(魔法方法)
魔法方法是 Python 类中前后双下划线包裹的特殊方法（如 __init__），会在特定场景自动调用，核心作用是 “自定义类的行为”
#### 举例
##### 1.`__init__`：对象初始化
1.作用：创建对象时自动执行，给对象绑定初始属性；
2.注意：第一个参数必须是 self
##### 2.`__str__`：打印自定义对象
作用：用 print(对象) 或 str(对象) 时，返回自定义字符串（默认显示内存地址）
```
class Student:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"学生对象：姓名={self.name}"

stu = Student("Alice")
print(stu)  # 输出：学生对象：姓名=Alice（而非默认的 <__main__.Student object at 0x...>）
```
##### 3.`__add__`：将自定义对象相加
作用：让两个对象可以用 + 运算，返回自定义结果
```
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):  # other 是另一个对象
        return Point(self.x + other.x, self.y + other.y)  # 返回新对象

p1 = Point(1, 2)
p2 = Point(3, 4)
p3 = p1 + p2  # 自动调用 __add__ 方法
print(f"({p3.x}, {p3.y})")  # 输出：(4, 6)
```
##### 4.`__len__`：计算自定义对象的长度
作用：用 len(对象) 时，返回自定义的 “长度”
```
class Student:
    def __init__(self, scores):
        self.scores = scores

    def __len__(self):
        return len(self.scores)  # 成绩列表的长度即科目数

stu = Student([90, 85, 95])
print(f"科目数：{len(stu)}")  # 输出：3（自动调用 __len__）
```
##### 5.`__repr__`：交互式环境下显示对象
作用：在 Python 交互模式下直接输入对象名，返回更详细的调试信息（与 __str__ 互补）
```
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Student(name='{self.name}', age={self.age})"

# 交互模式下输入 stu = Student("Bob", 19)，再输入 stu，会显示：Student(name='Bob', age=19)
```
## 四.文本处理：re正则表达式
### 简介
正则表达式是一种用于描述字符串模式的语法，搭配 Python re模块，能高效解决查找、提取、替换、验证文本等问题
### 1.正则核心语法（元字符 + 模式）
#### 1. 匹配 “单个字符”（基础元字符）
1..：匹配任意字符（除换行\n，加re.S修饰符可匹配换行）
2.[abc]：匹配 a、b、c 中的任意一个（字符集）
3.[^abc]：匹配除 a、b、c 外的任意字符（反向字符集）
4.\d：匹配数字（等价于[0-9]）
5.\w：匹配字母、数字、下划线（等价于[a-zA-Z0-9_]）
6.\s：匹配空白字符（空格、制表符\t、换行\n等）
7.\D/\W/\S：对应反向（非数字、非单词字符、非空白）
#### 2. 匹配 “数量”（量词）
1.*：前面的字符匹配 0 次或多次（贪婪：尽量多匹配）
2.+：前面的字符匹配 1 次或多次
3.?：前面的字符匹配 0 次或 1 次（可选）
4.{n}：前面的字符匹配恰好 n 次
5.{n,}：前面的字符匹配至少 n 次
6.{n,m}：前面的字符匹配 n 到 m 次
#### 3. 匹配 “边界”（定位符）
1.^：匹配文本开头（比如^abc仅匹配 “abc 开头” 的文本）
2.$：匹配文本结尾（比如abc$仅匹配 “abc 结尾” 的文本）
3.\b：匹配单词边界（比如\bapple\b仅匹配独立的 “apple”，不匹配 “pineapple”）
#### 4.运算
1.|：或运算
#### 5.分组与捕获（()）
用()包裹部分模式，可单独提取该部分内容，通过group(1)/group(2)获取（group(0)是整个匹配结果）
### 2.re 模块核心函数  
1.re.findall(pattern, text)
查找所有匹配的内容，返回列表
2.re.sub(pattern, repl, text)
替换匹配的内容，返回新文本
3.re.search(pattern, text)
全局查找第一个匹配项，返回匹配对象
4.re.match(pattern, text)
从文本开头匹配，返回匹配对象
5.re.split(pattern, text)
按匹配模式分割文本，返回列表
6.re.compile(pattern)
编译正则表达式，返回 Pattern 对象
```
import re

text = "我的手机号是13812345678，备用号13987654321，邮箱是abc123@qq.com"

# 1. findall：提取所有手机号（匹配11位数字，以13/14/15/17/18/19开头）
phone_pattern = r"1[3-9]\d{9}"  # 正则模式：1 + 3-9任意数字 + 9个数字
phones = re.findall(phone_pattern, text)
print("提取的手机号：", phones)  # 输出：['13812345678', '13987654321']

# 2. sub：打码手机号（中间4位替换为****）
phone_mask = re.sub(r"1[3-9](\d{4})(\d{4})", r"1\2****\3", text)  # 分组替换
print("打码后文本：", phone_mask)

# 3. search：查找第一个邮箱
email_pattern = r"[a-zA-Z0-9_]+@[a-zA-Z0-9]+\.[a-zA-Z]+"
email_match = re.search(email_pattern, text)
if email_match:
    print("找到的邮箱：", email_match.group())  # 输出：abc123@qq.com

# 4. compile：重复使用模式（比如多次匹配邮箱）
email_compile = re.compile(email_pattern)
text2 = "新邮箱xyz789@gmail.com，旧邮箱test@163.com"
emails = email_compile.findall(text2)
print("批量提取邮箱：", emails)  # 输出：['xyz789@gmail.com', 'test@163.com']
```
### 3.使用技巧
#### 1.贪婪 vs 非贪婪匹配：默认贪婪（尽量多匹配），加?变为非贪婪（尽量少匹配）
```
text = "<h1>标题1</h1><h1>标题2</h1>"
# 贪婪匹配（默认）：会匹配整个字符串（从第一个<h1>到最后一个</h1>）
greedy = re.findall(r"<h1>.*</h1>", text)
# 非贪婪匹配：加?，匹配到第一个</h1>就停止
non_greedy = re.findall(r"<h1>.*?</h1>", text)
print(greedy)  # 输出：['<h1>标题1</h1><h1>标题2</h1>']
print(non_greedy)  # 输出：['<h1>标题1</h1>', '<h1>标题2</h1>']
```
#### 2.修饰符简化匹配：通过flags参数设置，如re.I（忽略大小写）
```
text = "Apple apple APPLE"
# 忽略大小写匹配所有apple
apples = re.findall(r"apple", text, flags=re.I)
print(apples)  # 输出：['Apple', 'apple', 'APPLE']
```
#### 3.转义特殊字符：正则中\、.、*等是元字符，匹配原字符需加\转义（或用原始字符串r""）
```
text = "价格：99.9元"
# 匹配小数点（.需转义）
price = re.findall(r"\d+\.\d+", text)
print(price)  # 输出：['99.9']
```
## 五.代码格式：列表推导式、Type Hint（类型注释）
### 1.列表推导式 
列表推导式的核心是用紧凑格式替代冗余的 for 循环 + append，用一行搞定批量处理
#### 1.基础 
语法：[表达式 for 变量 in 可迭代对象 if 条件]
```
nums = [1,2,3,4,5]
even_squares = [x**2 for x in nums if x % 2 == 0]
```
#### 2.嵌套 
```
nested = [[[1,2], [3,4]], [[5,6], [7,8]]]
bad = [num for row  in matrix  for num in row if num % 2 == 0]
```
外层循环：for row in matrix
matrix 是二维列表 [[1,2], [3,4], [5,6]]，所以 row 会依次取到子列表 [1,2]、[3,4]、[5,6]
内层循环：for num in row
对于每个 row（比如 [1,2]），num 会依次取到子列表里的元素 1、2（同理，[3,4] 对应 3、4；[5,6] 对应 5、6）
#### 3.延深：字典 / 集合推导式
```
# 字典推导式（格式：{键表达式: 值表达式 for ... if ...}）
names = ["Alice", "Bob"]
name_len = {name: len(name) for name in names if len(name) > 3}

# 集合推导式（格式：{表达式 for ... if ...}，自动去重）
nums = [1,2,2,3,3,3]
unique_squares = {x**2 for x in nums}
```
### 2.Type Hint（类型注释）
Type Hint 可以标注变量 / 函数的类型，便于读者理解
#### 1.变量的类型注释
格式：变量名: 类型 = 值
```
age: int = 18
name: str = "Alice"
is_student: bool = True
score: float = 90.5
scores: list[int] = [90, 85, 95]  # 列表，元素都是int
student: dict[str, int] = {"Alice": 90, "Bob": 85}  # 字典，键str，值int
```
#### 2.函数的类型注释
格式：def 函数名(参数: 类型) -> 返回值类型: 
```
# 1：简单函数
def add(a: int, b: int) -> int:
    return a + b

# 2：结合容器类型
def get_avg(scores: list[float]) -> float:
    return sum(scores) / len(scores)

# 3：可选类型（允许None，需from typing import Optional）
from typing import Optional

def find_name(names: list[str], target: str) -> Optional[str]:
    """找到返回名字，没找到返回None"""
    return target if target in names else None
```
## 六.进阶技巧：generator生成器（yield关键字）
生成器的本质是可迭代对象（支持 for 循环遍历），通过yield关键字实现 “暂停 - 恢复” 逻辑，核心优势是不一次性占用大量内存（按需生成数据）
### 1.创建方式
#### 1.生成器表达式
格式和列表推导式几乎一致，把[]换成()，直接创建生成器，无需写函数
```
# 列表推导式（一次性生成所有数据，占内存）
list_nums = [x*2 for x in range(1000000)]  # 生成100万条数据，直接占内存

# 生成器表达式（按需生成，仅占迭代器本身内存）
gen_nums = (x*2 for x in range(1000000))

# 验证：生成器需通过迭代获取数据（如for循环、next()）
print(next(gen_nums))  # 输出：0（生成第1条）
print(next(gen_nums))  # 输出：2（生成第2条）
for num in gen_nums:
    if num > 10:
        break
    print(num)  # 输出：4,6,8,10（按需生成，不浪费内存）
```
#### 2.yield 关键字（自定义生成器函数）
在函数中用yield替代return，函数就变成生成器，调用时不执行函数体，返回生成器对象，每次迭代触发 “暂停 - 恢复”
```
def my_generator(n):
    for x in range(n):
        yield x*2  # 暂停函数，返回当前值；下次迭代从这里恢复
        print(f"恢复后，继续执行（x={x}）")

# 调用生成器函数：不执行代码，返回生成器对象
gen = my_generator(3)

# 迭代获取数据（3种方式）
print(next(gen))  # 输出：0 → 暂停在yield处
print(next(gen))  # 输出： 恢复后，继续执行（x=0）→ 2 → 再次暂停
```

```
def my_generator(n):
    for x in range(n):
        yield x*2  # 暂停函数，返回当前值；下次迭代从这里恢复
        print(f"恢复后，继续执行（x={x}）")
# 调用生成器函数：不执行代码，返回生成器对象
gen = my_generator(3)        
for num in gen:
    print(num)  
```
输出:
```
      0
      恢复后，继续执行（x=0）
      2
      恢复后，继续执行（x=1）
      4
      恢复后，继续执行（x=2）
```
yield 工作原理：像 “暂停键 + 返回值”—— 执行到yield时，返回值并暂停函数状态，下次迭代（next ()/for 循环）从暂停处继续执行，直到函数结束
#### 3.yield from（简化嵌套生成器）
多层生成器嵌套时，用yield from直接迭代内层生成器
```
def gen1():
    yield 1
    yield 2

def gen2():
    yield from gen1()  # 直接迭代gen1的生成器
    yield 3

for num in gen2():
    print(num)  # 输出：1,2,3
```