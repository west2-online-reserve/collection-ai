# 接下来我将整理python基础语法和总结技巧

## str

- 我整理了一些字符串的方法

```python
text = "   Hello World   "
#/删去开头结尾的空格
text.strip()
#/全部大写
text.upper()
#/全部小写
text.lower()
#/替换字符，把前替换成后
text.replace("h","j")
#/将“ ”为分割符把字符串变成列表，如果括号为空，则以空白字符为分割符
text.split(" ")
#检查开头或索引的字符串内容返回bool值
text.startswith('...',index)
```

## List

- 列表是 python 的一种数据类型
- 用来存储一些有序的数据并用于调用这些数据
- 我整理了一些列表的方法

```python
list = [1,2,54,7,4]
#加入某个元素
list.append()
#连接两个列表，其实我感觉 list_1 += list_2 也实现？
list.extend()
#可以在特定索引处插入某元素
list.insert(0,1)
#移除并返回索引处元素
list.pop()
#移除某元素
list.remove()
#获取列表元素的索引和索引内容
for x,y in enumerate(list):
#获得倒着的列表
reversed(list)
```

## Lambda

- labda 的语法

```python
#labda 是用于定义 lambda 函数；argument 是参数列表，冒号后是表达式
lambda arguments :expression
#lambda 一般搭配 map() , filter() ,reduce() 一起使用，其中除 reduce 是 functools 库函数以外，其余为内置函数。

#map(func(),list) 相对于映射，把第二个参数，即列表或元组的数据依次输入 func() ,并把返回结果存入内存，类型为 map。
a = list(map(lambda x:x**2,[1,2,3,4,5]))

#filter(func(),list) 的输入参数类似 map()，但 func() 的放回结果为 bool 值，为 True 则返回其值
b = list(fiter(lambda x,x%2 == 0),[1,2,3,4,5,6])

#关于 reduce(func(),list) ，func（x,y）里要输入两个参数，有点像 复合函数 func(func(x1,x2)x3)
c = reduce(lambda x,y:x*y,[1,2,3,4,5,6])
```

## dict

- dict.keys() 生成的是 dict_keys 类型，而非列表，要让它成为列表，需要 list() 。

## 正则表达式

- 使用正则表达式需要引用 re 库

## 列表推导式

- 举例
  - 传统写法

    ```python
    squares = []
    for x in range(5):
        squares.append(x * x)
    ```

  - 列表推导式写法

    ```python
    squares = [x * x for x in range(5)]  # 结果: [0, 1, 4, 9, 16]
    ```

## 类型注释

- 举例

```python
def greet(name: str, age: int) -> str:
    return f"你好 {name}, 你今年 {age} 岁"
#（name: str 表示 name 应该是字符串，-> str 表示函数返回字符串）
```

## 生成器

- 用于节省内存

## 装饰器
