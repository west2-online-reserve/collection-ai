import time
def decorate(func):
    def wrap(*args,**kwargs):
        print(f"Function_name : {func.__name__}")
        start_time = time.time()
        ret = func(*args,**kwargs)
        end_time = time.time()
        process_time = end_time - start_time
        print(f"{func.__name__} 于 {start_time} 开始运行")
        print(f"{func.__name__} 于 {end_time} 结束运行")
        print(f"{func.__name__} 运行了 {process_time}")
        return ret
    return wrap
@decorate
def square(x):
    return x**2

print(square(5))

