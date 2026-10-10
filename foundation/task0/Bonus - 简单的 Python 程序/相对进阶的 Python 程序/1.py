# 实现一个装饰器
# 在开始执行函数时输出该函数名称，
# 并在结束时输出函数的开始时间和结束时间以及运行时间


import random
import time
from datetime import datetime

# 装饰器
def record_time(func):
    def inner(*args,**kwagrs):
        t1 = datetime.now()
        func(*args,**kwagrs)
        t2 = datetime.now()
        print(t1)
        print(t2)
        return t2-t1
    return inner

# 执行函数
# @record_time
def test():
    print("函数开始执行")
    print("waiting......")
    time.sleep(random.randint(1,10))
    print("函数执行结束")

# 测试
test = record_time(test)
print('-'*35)
print(test())