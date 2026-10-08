import time

def time_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"开始执行函数: {func.__name__}")
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        run_time = end - start
        print(f"函数开始时间: {start}")
        print(f"函数结束时间: {end}")
        print(f"运行时间: {run_time:.6f} 秒")
        return res
    return wrapper



@time_decorator
def test_func(n):
    s = 0
    for i in range(n):
        s += i
    return s

test_func(1000000)
