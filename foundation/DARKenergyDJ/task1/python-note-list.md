# python note

## list

基本概念: 列表是一种序列有序：元素按插入顺序保存，可通过索引访问。

可变：可以增、删、改。

可重复：允许出现相同元素。

可存储任意类型：整数、字符串、列表、对象等都可以

### 列表的增，删，改，查

## 增加元素

|方法|说明|
|---|---|
|append(x)|末尾添加一个元素|
|extend(iterable)|末尾扩展多个元素|
|insert(i, x)|在索引 i 处插入 x|

## 减少元素

|方法|作用|
|---|---|
|del lst[i]|删除指定索引元素|
|pop(i)|删除并返回索引 i 的元素，默认末尾|
|remove(x)|删除第一个值为 x 的元素|
|clear()|清空列表|

## 改动元素

列表名[下标] = "元素名"

或者使用切片语法

## 查询元素

lis.index() 查询指定元素的下标
lis.count() 查询指定元素出现的次数
len(lis) 查询列表长度

## 列表推导式

[expression for target in iterable]

将可迭代对象的每一个元素存入target然后进行express处理

列表推导式也可以进行嵌套：a = [col for row in matrix for col in row]

    for row in matrix:

        for col in row:
先外后内

## 列表排序

lis.sort()正序排序
lis.reverse()到序排序

## copy

浅拷贝：list.copy() 或者使用切片    浅拷贝只拷贝外层，嵌套对象仍然共享

深拷贝：import copy
lis2 = copy.deepcopy(a)
