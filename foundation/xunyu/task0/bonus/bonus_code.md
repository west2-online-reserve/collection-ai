# bonus code

## 1. 输入三个整数 x、y、z，用多种方式由大到小输出

```python
# 输入三个整数x、y、z，尝试用多种方式把这三个数由大到小输出

x = int(input("请输入第一个数x："))
y = int(input("请输入第二个数y："))
z = int(input("请输入第三个数z："))


list1 = [x, y, z]
list1.sort(reverse=True)
print(list1)

print(sorted([x, y, z], reverse=True))
```

## 2. 输出九九乘法表

```python
# 九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}*{i}={i * j}", end="\t")
    print()
```

## 3. 判断字符串是否含有"ol"并替换再倒序

```python
# 输入一个字符串，判断是否含有"ol"这个子串，
# 若有则把所有的"ol"替换为"fzu"，最后把字符串倒序输出

s = input("请输入一个字符串：")
if "ol" in s:
    print("含有")
    s = s.replace("ol", "fzu")
else:
    print("不含")

print(s[::-1])  # 倒序输出
```

## 4. 删除列表中的字符串元素再排序

```python
# 输入一个列表，列表中含有字符串和整数，
# 删除其中的字符串元素，把剩下的整数升序排序，输出列表

lst = eval(input())
new_lst = []

for item in lst:
    if type(item) == int:
        new_lst.append(item)

new_lst.sort()
print(new_lst)
```

## 5. 字典操作：删除学号尾号为偶数的元素

```python
# 创建一个字典，键为学号，值为姓名，
# 删除学号尾号为偶数的元素，输出字典

d = {}
d["202101"] = "张三"
d["202102"] = "李四"
d["202103"] = "王五"
d["202104"] = "赵六"
d["202105"] = "孙七"

print("删除前：", d)

# 注意：遍历字典的时候不能直接删，要先取出来
keys = list(d.keys())
for k in keys:
    if int(k[-1]) % 2 == 0:  # 判断学号最后一位是不是偶数
        del d[k]

print("删除后：", d)
```

## 6. 统计列表中各个数字出现的次数

```python
# 创建一个函数，统计一个只有数字的列表中各个数字出现的次数，通过字典方式返回

def count_number(lst):
    result = {}
    for num in lst:
        if num in result:
            result[num] = result[num] + 1
        else:
            result[num] = 1
    return result

print(count_number([1, 2, 2, 3, 3, 3, 1, 4, 5, 5]))
# 输出: {1: 2, 2: 2, 3: 3, 4: 1, 5: 2}
```

## 7. 设计一个商品类

```python
# 设计一个商品类
# 私有数据成员：商品序号、商品名、单价、总数量、剩余数量
# 公有成员函数：init（初始化）、display（显示信息）、income（计算已售出商品价值）、setdata（修改商品信息）

class Goods:
    def __init__(self, num, name, price, count, left):
        self.__num = num        # 商品序号
        self.__name = name      # 商品名
        self.__price = price    # 单价
        self.__count = count    # 总数量
        self.__left = left      # 剩余数量

    def display(self):
        print(f"商品序号:{self.__num} 商品名:{self.__name} 单价:{self.__price} 总数量:{self.__count} 剩余数量:{self.__left}")

    def income(self):
        # 已售出商品价值 = 单价 × 已售数量（总数量 - 剩余数量）
        sold = self.__count - self.__left
        return sold * self.__price

    def setdata(self, num, name, price, count, left):
        self.__num = num
        self.__name = name
        self.__price = price
        self.__count = count
        self.__left = left

g = Goods("001", "苹果", 5, 100, 30)
g.display()
print("已售出商品价值：", g.income())

g.setdata("002", "香蕉", 3, 50, 10)
g.display()
print("已售出商品价值：", g.income())
```

## 8. 斗地主随机发牌（不要求花色）

```python
# 斗地主随机发牌，把三个玩家的牌和多的三张牌分别写入4个文件
# 用0~53代表54张牌，不要求花色

import random

cards = list(range(54))  # 生成54张牌
random.shuffle(cards)    # 洗牌

player1 = cards[0:17]
player2 = cards[17:34]
player3 = cards[34:51]
others = cards[51:54]   # 底牌三张

def write_to_file(filename, hand):
    with open(filename, "w", encoding="utf-8") as f:
        for card in hand:
            f.write(str(card) + " ")

write_to_file("player1.txt", player1)
write_to_file("player2.txt", player2)
write_to_file("player3.txt", player3)
write_to_file("others.txt", others)

print("发牌完成！")
```

## 9. 进阶：装饰器记录函数运行时间

```python
# 实现一个装饰器：开始执行函数时输出该函数名称，
# 结束时输出函数的开始时间和结束时间以及运行时间

import time

def my_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()  # 开始时间
        print(f"开始执行函数：{func.__name__}")
        result = func(*args, **kwargs)
        end = time.time()    # 结束时间
        print(f"函数开始时间：{start}")
        print(f"函数结束时间：{end}")
        print(f"函数运行时间：{end - start}")
        return result
    return wrapper

@my_decorator
def test():
    s = 0
    for i in range(1000000):
        s = s + i
    return s

test()
```

## 10. 斗地主发牌（按从大到小排序输出）

```python
# 斗地主随机发牌，把每个人的牌和多的三张牌
# 按照从大到小的顺序输出到 player1.txt、player2.txt、player3.txt、others.txt

import random

cards = list(range(54))
random.shuffle(cards)

# 分牌并排序（从大到小）
player1 = sorted(cards[0:17], reverse=True)
player2 = sorted(cards[17:34], reverse=True)
player3 = sorted(cards[34:51], reverse=True)
others = sorted(cards[51:54], reverse=True)

with open("player1.txt", "w", encoding="utf-8") as f:
    for c in player1:
        f.write(str(c) + " ")

with open("player2.txt", "w", encoding="utf-8") as f:
    for c in player2:
        f.write(str(c) + " ")

with open("player3.txt", "w", encoding="utf-8") as f:
    for c in player3:
        f.write(str(c) + " ")

with open("others.txt", "w", encoding="utf-8") as f:
    for c in others:
        f.write(str(c) + " ")

print("发牌完成！")
```

## 11. 列表推导式生成矩阵并转置

```python
# 用列表推导式生成一个5*10的矩阵，所有值为1
matrix = [[1 for j in range(10)] for i in range(5)]
print(matrix)

# 再用列表推导式把这个矩阵转置（10*5）
transpose = [[matrix[i][j] for i in range(5)] for j in range(10)]
print(transpose)
```

## 12. 类的魔术方法：MyZoo

```python
# 创建类 MyZoo，实现以下功能：
# 1. 具有字典 animals，动物名称作为key，动物数量作为value
# 2. 实例化对象的时候，输出"My Zoo!"
# 3. 创建对象时可以用字典初始化，无字典则初始化animals为空字典
# 4. print(myzoooo) 输出动物名称和数量
# 5. 比较两个对象时，只要动物种类一样就判断相等
# 6. len(myzoooo) 输出所有动物总数

class MyZoo:
    def __init__(self, animals=None):
        if animals is None:
            self.animals = {}
        else:
            self.animals = animals
        print("My Zoo!")

    def __str__(self):
        return str(self.animals)

    def __eq__(self, other):
        # 只要动物种类一样就相等，数量不管
        return set(self.animals.keys()) == set(other.animals.keys())

    def __len__(self):
        # 所有动物总数
        total = 0
        for n in self.animals.values():
            total = total + n
        return total

myzoooo = MyZoo({"pig": 5, 'dog': 6})
myzoooo = MyZoo()
print(myzoooo)

myzoooo1 = MyZoo({'pig': 1})
myzoooo2 = MyZoo({'pig': 5})
print(myzoooo1 == myzoooo2)  # True，种类一样就相等
print(len(myzoooo1))          # 输出: 1
print(len(myzoooo))           # 输出: 0
```

## 13. 正则表达式验证密码

```python
# 写一个正则表达式，用于验证用户密码
# 长度在6~18之间，只能包含英文和数字

import re

pattern = r'^[A-Za-z0-9]{6,18}$'

password = input("请输入密码：")
if re.match(pattern, password):
    print("合法")
else:
    print("不合法")
```
