# 函数

## 基本语法

本质是可以重复调用的代码块

基本结构如下：

    def myfunc(参数):
        #内容
        return 返回值
调用时：

函数名加()

## 参数

## 位置参数

位置参数
按顺序传递

    def sub(a, b):
        return a - b

    sub(5, 3)  # 2

## 关键字参数

按参数名传递，顺序可以不同

    sub(b=3, a=5)  # 2

## 默认参数

定义时给默认值，调用时可省略

## 可变位置参数 *args

接收任意个位置参数，打包成元组。

    def add_all(*args):
        return sum(args)

    add_all(1, 2, 3)  # 6

## 可变关键字参数 **kwargs

接收任意个关键字参数，打包成字典。

    def show(**kwargs):
        print(kwargs)

    show(name="Tom", age=18)  # {'name': 'Tom', 'age': 18}

注意位置参数必须写在关键字参数之前

## 嵌套函数和闭包

函数可进行嵌套 一般来说变量依据LEBG规则进行操作

local>enclose>global>build-in

可以使用nonelocal语句对变量进行声明，使得内层变量可以修改外部环境和函数

当内层变量作为外部函数的返回值，应用外层函数变量，存入一个变量时，会产生闭包

    def power():
        def exp-of(base):
            return base**exp
        return exp_of
    square = power(2)
    cube = power(3)
利用nonelocal 能使得内层函数修改外层函数的特性，可以实现简单的记忆函数
