import time

def my_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

@my_decorator       #这个语法糖我参考ai的建议，特此声明（学到知识是我的任务 = 我的包装器）
def test_task():
    time.sleep(1.5)
    print("原函数内部的代码执行完毕！")
    return "这是测试函数的返回值"

print("---开始执行测试任务---")
test_result = test_task()
print(f"测试任务的返回值: {test_result}")
print("---测试任务执行完毕---")