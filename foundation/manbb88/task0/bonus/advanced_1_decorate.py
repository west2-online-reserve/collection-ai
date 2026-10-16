import time as t
from functools import wraps

def my_dec(func):
    @wraps(func)
    def inner(*args, **kwargs):
        print(f"开始执行{func.__name__}函数")
        start=t.time()
        result = func(*args, **kwargs)
        end=t.time()
        elapsed = end - start
        start = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(start))
        end = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(end))
        print(f"函数{func.__name__}执行开始时间为{start},结束时间为{end},总共耗时{elapsed}秒")
        return result
    return inner