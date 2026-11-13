#一个装饰器，用来实现计算函数运行时长
import time
from functools import wraps

def timer(func):
    @wraps(func)    #防止函数名被替换,比如原函数名叫add,但是包装函数里叫做wrapper
    def wrapper(*args, **kwargs):
        print(f"开始执行函数: {func.__name__}")

        start_time = time.time()
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"开始时间: {start_str}")

        #执行原函数
        result = func(*args, **kwargs)

        end_time = time.time()
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"结束时间: {end_str}")

        #计算运行时间
        run_time = end_time - start_time
        print(f"运行时间: {run_time:.4f}秒")

        return result

    return wrapper

@timer #时间装饰器
def add(a, b):
    time.sleep(1)   #防止运行时间过短而输出不可见
    return a + b

print("add的结果:", add(3, 5))