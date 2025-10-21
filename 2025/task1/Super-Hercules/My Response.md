- 数据结构列表、字典的使用
    列表:
        1.使用中括号创建列表，每个元素用逗号隔开。
        2.使用+可以合并两个列表。
        3.list.append()在列表尾部增加元素。
        4.list[i] = "*"修改i位置上的元素。
        5.list.insert(i, *)在i位置上插入元素。
        6.list.remove("*")删除元素。
        7.list.pop()弹出最后一个元素，也即删除并输出最后一个元素。常用于栈？
        8.list[开始:终点:步长]列表切片操作。
        9.list.sort()排序。
        10.list.reverse()反转列表操作。
        11.list.index(*)查找元素的索引。

    字典：字典用于储存无序的键与值的指向关系，可以快速查找需要的元素
        1.使用花括号创建字典或者dict = dict(*, *)
        2.dict[key]返回字典中该键对应的值。
        3.del dict[key]删除字典中该键对应的键值对。
        4.使用in查询一个键是否存在于字典中，为白话文，即 key in dict，返回布尔值。
        5.dict.get(key, default)安全获取该键对应的值，不存在则返回默认值，不会引发错误。
        6.dict.pop(key)弹出并删除。
        7.用for key in dict来遍历字典中每一个键。
        8.用for key, value in dict.items()遍历字典中每一个键值对。
        9.{x: expression}字典推导式。
        10.dict1 | dict2合并字典。

- Lambda匿名函数
    它用于创建小巧的匿名函数，即不需def定义与声明函数名的函数

```python
lambda arguments: expression
```

它会返回表达式的运行结果。

- Decorator装饰器
    它能够给函数增加一些（聊胜于无？）的特效且不改变函数本身，格式如下：

```python
def decorator(func):
    def wrapper():
        exprssion
        func()
        exprssion
    return wrapper

@decorator
def function():
    expression

function()
```

- 类Class、Magic Methods的使用
    类：是对象的模板，定义了对象具有什么样的属性，能够使用什么样的方法。

```python
class Men:      #👇第一个参数固定表示对象本身，不需要手动传入。
    def __init__(self, name, age):
        self.name = name    #这些操作将属性绑定在对象身上。
        self.age = age
ZhangSan = Men("张三", 18)  #实例化一个名为张三、年岁18的人类对象。
print(ZhangSan.name)   #输出实例ZhangSan名为name的属性，即"张三"。
```

Magic Methods：
        魔法方法定义了类要如何处理运算符和内置函数，让我们更好地控制类和对象的行为。最常用的

```python
__init__()
```

   构造函数即在类中定义了类的对象，它会在对象实例化时自动调用。

```python
__del__()
__str__()
__add__()
```

   这些魔法方法能够在实例回收、打印、做加法等操作时触发设定好的内容。

- re正则表达式的使用
    在Python 3.13.9 文档中，正则表达式是这样被表述的：
&gt;正则表达式（或 RE）指定了一组与之匹配的字符串；模块内的函数可以检查某个字符串是否与给定的正则表达式匹配（或者正则表达式是否匹配到字符串，这两种说法含义相同）。
    使用正则表达式需要进口

```python
import re
```

   请允许我使用进口这个翻译，它在我早期的学习过程中起到了重要的作用。

   正则表达式的一般语法结构：

```python
import re  #进口。
pattern = re.compile(r"re expression")  #将正则表达式作为一个匹配的模式存入对象pattern。
for a in pattern.findall(content):  #在对象content中寻找全部的符合正则表达式的文本。可以看出，findall返回一个列表。
    #print(type(a))  #如打印a的类型，得到的结果就是字符串（str）。
    print(a)  #打印。
```

   正则表达式本身通常与一般的字符串没有什么区别，它们只是正好符合正则表达式的语法而已，因此在正则表达式里的一般字符可以在目标文本中匹配到它本身。而在语法中规定结构的特殊字符，即元字符（MetaCharacter），有：.、^、$、*、+、?、{}、[]、\、|和()。
        .：
            通配符，可以匹配任何字符，但不包括换行符。

*：
            表示匹配前面的子表达式任意次，包括0次。是什么意思呢？就是表示在它前面那个字符可以匹配任意次数，比如我想从

```python
menu = """
至尊炸猪蹄炸蛋超辣螺蛳粉
老友粉
桂林米粉
甜甜的非糯玉米
干捞粉
乌骨鸡
羊肉粉
羊杂粉
粉利
冒烤鸭
素汤切粉
荷叶烤鸡
烧鸭粉
脆皮猪五花
牛杂粉
雪花和牛
白切鸡
顶级荔浦芋头扣肉
梅菜扣肉
韩式无骨炸鸡块
普通螺蛳粉
牛腩粉
狗肉粉
痛痛粉
酸辣粉
三鲜粉
"""
```

   中寻找所有的米粉，我就可以用

```python
re.compile(r".*粉")    
```

   在menu的每行中寻找包含“粉”的词，且取回它前面的每一个字。

   +：
    与*有一点点区别，它前面的字符必须至少出现一次。

   {}：
    表示匹配次数，要求字符必须连续出现几次。例如，狐{2}表示狐这个字符应该连续出现两次；猪{3, 4}表示猪这个字符至少连续出现三次，至多连续出现四次。

   []：
    方括号表示要匹配指定的几个字符之一。如[abc123]可以匹配a、b、c、1、2、和3；[a-c]可以匹配从a到c的三个字母。
    此外，一些元字符在方括号内会被转义为普通字符，如.。
    如果在方括号中使用^，表示非方括号里面的字符集合。如[^\d]表示全部非数字。

   \：
    1.当我们将.、*和?等元字符输入正则表达式并希望它匹配文本中的相应符号时，我们会发现它们却作为元字符生效了，这时在元字符的前面加上\进行转义，它将作为一个普通字符正常匹配。
    2.
   \+一些字符，可以匹配特定类型的字符。

   \d：匹配0-9之间任意一个数字字符。

   \D：匹配任意一个非0-9数字的字符。

   \s：匹配任意一个空白字符，包括 空格、tab、换行符等。

   \S：匹配任意一个非空白字符。

   \w：匹配任意一个文字字符，包括大小写字母、数字、下划线。
        缺省情况也包括 Unicode文字字符，如果指定 ASCII 码标记，则只包括ASCII字母。将compile的第二个参数设置为re.A或re.ASCII即可。
        例如：

```python
            re.compile(r"\w{2, 4}", re.A)
```

   compile的第二个参数亦可以设置为其他规则。

   \W：匹配任意一个非文字字符。

   反斜杠也可以用在方括号里面，比如 [\s,.] 表示匹配：任何空白字符，或者逗号，或者点。

|：
    跟方括号类似，不过它可以匹配分隔开的每一个整体的字符串。比如，AAA FZU|桂林三金药业|武宣县大米厂有限公司。

():
    括号称为正则表达式的组选择。组就是把正则表达式匹配的内容其中的某些部分标记为某个组。
    比如我们有一个壮语单词表：

```python
    list = """
        堵木，猪猪
        堵茨，牛牛
        堵拐，蛙蛙
        堵bia，鱼鱼
        谢乜梦，***
        卖乜梦，***
        被骂，回家
        供厄样，吃饭
    """
```

   我们希望将拼读的壮语词汇挨个拎出来，可以用逗号来作为标识，可是并不需要逗号，这时候就轮到括号出场了。
   我们使用神奇的

```python
   re.compile("^(.*)，",re.M)
```

就能把每行的逗号之前的每一个字分入括号的组里啦。只有一个分组的情况下，输出的是括号内的文本。
    另外，分组还可以多次使用并给每个组别命名以方便提取特定内容。格式如下：

```python
import re
    pattern = re.compile(r"^(?P<分组名1>re expression)，.+ (?P<分组名2>re expression)")
    for match in pattern.finditer(content):
        print(match.group('分组名1'))
        print(match.group('分组名2'))
```

贪婪模式与非贪婪模式以及?：
    在正则表达式中“*”“+”“?”都是“贪婪的”，它们会匹配尽可能多的字符。例如使用

```python
re.compile(r"<.*>")
```

   匹配

```python
tag = "<html><head><title>Title</title>"
```

   它会取到整串字符串，而不是每一个小括号分别返回。
    这时我们增加一个?，即<.*?>它将会尽可能少地匹配字符，把每一个小括号及其内容分别输出。

^：
    除了方括号中表达“非”的作用，它还可以表示匹配文本的开头位置。正则表达式可以设定单行模式和多行模式。如果是单行模式，^表示匹配整个文本的开头位置。如果是多行模式，则表示匹配文本每行的开头位置。将compile的第二个参数设置为re.M启用多行模式。
    下面的文本中，每行最前面的数字表示水果的编号：

```python
menu = """
    001-沃柑-1.5/斤
    002-砂糖橘-0.7/斤
    003-西瓜-0.68/斤
"""
```

   如果我们要提取所有的水果编号，用这样的表达式：

```python
fruit = re.compile("^\d+", re.M)
```

   如果不设置第二个参数，结果将只有001。

   $：跟^一样，不过它表示文本的结尾位置。

   正则表达式的split方法：

   &gt;“re.split(pattern, string, maxsplit=0, flags=0) 用pattern分开string。”

   比如我们是柳州某高级中学的秘书，需要从一份混乱的名单里筛选出一些学生的名字：

```python
list = "覃飞，韦廉；农艳春.张秀菊;黄桂七,李杜茨:蓝十二   莫香桃"
```

   可以看到每个名字之间的间隔符号各不相同，因此我们需要一个统一的标准将每一个名字完整地取出来：

```python
namelist = re.split(r"[;,，；:\.\s]\s*", list)
print(namelist)

['覃飞', '韦廉', '农艳春', '张秀菊', '黄桂七', '李杜茨', '蓝十二', '莫香桃']
```

   结果很漂亮啊。

   正则表达式的sub方法与match类：
   sub方法用于以一个函数的返回值替换掉在文本中匹配到的文本。

```python
import re

source = """
下面是这学期要学习的课程：

<a href='https://www.bilibili.com/video/av66771949/?p=1' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是牛顿第2运动定律

<a href='https://www.bilibili.com/video/av46349552/?p=125' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是毕达哥拉斯公式

<a href='https://www.bilibili.com/video/av90571967/?p=33' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是切割磁力线
"""

# 替换函数，参数是 Match对象
def subfunc(match):
    # Match对象 的 group(0) 返回的是整个匹配上的字符串。
    src = match.group(0)

    # Match对象 的 group(1) 返回的是第一个group分组的内容。
    number = f"/av{int(match.group(1)) + 6}/"

    # 返回值就是最终替换的字符串
    return number

newStr = re.sub(r"/av(\d+)/", subfunc , source)
print(newStr)
```

- 列表推导式
    又称列表解析式，用于简（偷）明（懒）扼（装）要（13）地创建列表。
    主要结构：

```python
    list = [argument for argument in iterable if expression]
```

- generator生成器（yield关键字）


- OOP面向对象编程思想
    面向对象编程思想在字面上可能不是很好理解，我更愿意称其对象导向的编程思想。它与面向过程编程思想的区别在于不再将项目编写为一个“任务说明书”，而是如同客观世界一样将任务涉及到的一切“主体”视作类定义下的对象。类同时定义了对象的属性和方法，即为其特征与能够做到的事。想要调用对象的属性只需要调用对象的名字及属性的名字，而不需要再次传入该参数。想要进行操作就调用方法，简便快捷。就像史书中的“纪传体”一样，《史记》十二本纪、三十世家、七十列传、十表和八书，每一篇对应一个人或一群人，让读者一眼盯真。例如，即使看不到封装了的方法里面到底有什么，我们也能知道它的作用。类也可以通过继承父类的对象及其属性方法，减少代码的重复性。此外，不同的子类能灵活处理大同小异的任务。

- Type Hint类型注释
示例：

```python
   def select_dialogues_by_affinity(self, dialogue_box: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.affinity < 30:
            result: List[Dict[str, Any]] = []
            for dialogue in dialogue_box:
                if not dialogue.get("high_affinity_only", False):
                    result.append(dialogue)
            return result
        elif self.affinity > 70:
            return dialogue_box
        else:
            return dialogue_box
```

在该函数中，我们可以看到函数名后标注的参数之后使用了洋文冒号加数据类型的结构，这就是类型标注，它不会影响程序的行为，但是却能有效地告诉读者这里应该传入什么样的数据类型。而在函数之后以->链接的数据类型则表示函数返回结果的数据类型。
基本格式：

```python
def func(argument: Type, argument: Type[Type]) -> Type:
    expression                          #👆表示该容器内的元素的数据结构  
```

如果需要使一个参数能够同时传入两种数据类型，则需要进口Typing模块中的Union：

```python
from Typing import Union

def func(argument: Union[Type1, Type2], argument: Type[Type]) -> Type:
    expression
```