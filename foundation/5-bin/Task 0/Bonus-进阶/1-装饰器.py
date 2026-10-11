import time

def my_decorator(func):
    def wrapper(*args, **kwargs):
        #原函数执行前
        print(f"开始执行函数，函数名称为{func.__name__}")
        start_time = time.time()#返回时间戳
        start = time.localtime()#返回时间元组

        result = func(*args, **kwargs)#与wrapper保持一致

        #原函数执行后
        end_time = time.time()
        end = time.localtime()
        #注意引号问题
        #time.strftime将时间元组转换为'%Y-%m-%d %H:%M:%S'形式
        print(f"函数的开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', start)}")
        print(f"函数的结束时间：{time.strftime('%Y-%m-%d %H:%M:%S', end)}")
        print(f"函数的运行时间：{end_time - start_time}s")

        return result
    return wrapper

@my_decorator
def ApulsB():
    a, b = map(int, input().split())
    print(a+b)

def main():
    ApulsB()

if __name__ == "__main__":
    main()