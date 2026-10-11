import time
from datetime import datetime
def log_runtime(func):
    def wrapper(*args,**kwargs):
        start_time=datetime.now()
        start=time.time()
        print(f"开始执行函数:{func.__name__}")
        result=func(*args,**kwargs)
        end_time=datetime.now()
        end=time.time()
        print(f"结束执行函数:{func.__name__}")
        print(f"开始时间:{start_time}")
        print(f"结束时间:{end_time}")
        print(f"运行时间:{end-start:.6f}秒")
        return result
    return wrapper

def work(seconds):
    time.sleep(seconds)
    return "执行完成"
work=log_runtime(work)
print(work(1.5))