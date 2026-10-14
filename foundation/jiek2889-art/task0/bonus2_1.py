import time
import random


def decorator(func):
    def wrapper():
        print("开始执行函数")
        start_time = time.time()
        print(f"函数执行时间：{start_time}")
        func()
        end_time = time.time()
        print(f"函数执行结束时间：{end_time}")
        print(f"函数执行时间为: {end_time - start_time}秒")

    return wrapper


@decorator
def function():
    time.sleep(random.randint(1, 3))


function()
