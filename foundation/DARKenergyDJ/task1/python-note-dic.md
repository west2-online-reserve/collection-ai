# 字典

## 特点

* 键值对结构：key: value

* 键必须唯一，且必须是可哈希类型，如字符串、数字、元组。

* 值可以是任意类型。

* 可变：可以增、删、改

## 创建字典

* 利用dict()函数

d = dict(A = 0,B = 1)

d = dic([(A,0),(B,1),(C,2)])

* 直接创建

x = {"A" : "1","B" : "2"}
*使用键值对创建时不可给键添加引号*

* .fromkeys方法

d = dict.fromkeys(iterable,value)

## 字典操作

### 访问元素

d = {"a": 1, "b": 2}

d.get("a")

d["a"]

"a" in d        # True

in 判断的是键，不是值

### 增删改

d.update({"c": 3, "d": 4})
d.update(a=5)

del d["a"]              # 删除键 a
x = d.pop("b")          # 删除并返回 2

d.keys()    所有键，返回视图

d.values()  所有值，返回视图

d.items()   所有键值对，返回视图

## 遍历字典

### 遍历键

for key in d:
    print(key)

### 遍历值

for value in d.values():
    print(value)

### 遍历键值对

for key, value in d.items():
    print(key, value)

## 拷贝与字典推导式与列表类似

squares = {x: x * x for x in range(5)}
