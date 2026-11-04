import time

# 1. 定义装饰器
def logger(func):
    def wrapper():
        # 函数执行前
        print("开始执行函数:", func.__name__)
        start_time = time.time()
        print("开始时间:", time.ctime(start_time))
        
        # 执行原函数（无参数）
        func()
        
        # 函数执行后
        end_time = time.time()
        print("结束时间:", time.ctime(end_time))
        print("运行时间: {:.2f} 秒".format(end_time - start_time))
    
    return wrapper


# 2. 测试用例（函数不带参数）
@logger
def do_something():
    print("正在执行任务...")
    time.sleep(1.5)  # 模拟耗时


# 3. 调用被装饰的函数
do_something()