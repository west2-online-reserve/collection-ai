# 装饰器

装饰器本质上是一个可调用对象，它接收一个函数（或类），返回一个新的函数（或类），用于在不修改原代码的情况下增强功能。

    @decorator
        def func():
        pass
等价于：

    func = decorator(func)

注意使用*args 和 **kwargs
让装饰器可以包装任意参数的函数。

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
多个装饰器执行顺序：

包装顺序：从下到上

调用顺序：从上到下

    import time
    def time-master(func):
            def call-func:
            start = time.time()
            func()
            stop = time.time()
            print(f"time={stop-start}s")
        return call-func
    
    @time-master
    myfunc()

## lambda 表达式

lambda 用于定义简单的匿名函数

    f = lambda x: x + 1
    print(f(1))  # 2
等价于：

    def f(x):
        return x + 1
注意：lambda 只能写一个表达式，不能写多行语句。

lambda arg1,arg2......argN : expression

lambda 表达式可以作为列表的一个参数
