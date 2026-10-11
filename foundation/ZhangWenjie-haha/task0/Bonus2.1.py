# reference from deepseek
# 这个任务我是用 ai 辅助完成的，太多语法上的细节了 qwq

import time
from datetime import datetime
from functools import wraps


def log_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_dt = datetime.now()
        start_perf = time.perf_counter()

        print(f"开始执行函数：{func.__name__}")

        try:
            return func(*args, **kwargs)
        finally:
            end_dt = datetime.now()
            elapsed = time.perf_counter() - start_perf

            time_fmt = "%Y-%m-%d %H:%M:%S.%f"
            print(f"函数 {func.__name__} 开始时间：{start_dt.strftime(time_fmt)[:-3]}")
            print(f"函数 {func.__name__} 结束时间：{end_dt.strftime(time_fmt)[:-3]}")
            print(f"函数 {func.__name__} 运行时间：{elapsed:.6f}s")

    return wrapper


@log_runtime
def demo():
    pass
    return "end"


result = demo()
print(result)
