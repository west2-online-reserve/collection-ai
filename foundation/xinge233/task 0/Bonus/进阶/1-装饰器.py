# 1. 实现一个装饰器，在开始执行函数时输出该函数名称， 并在结束时输出函数的开始时间和结束时间以及运行时间
import time


def decoration(func):
    def HowLong():
        start = time.time()
        func()
        end = time.time()
        print(str(func.__name__))
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
        print(f"开始时间{start_time}")
        print(f"结束时间{end_time}")
        print(f"运行了{(end - start):.5f}秒")

    return HowLong


if __name__ == "__main__":

    @decoration
    def TimeToSleep():
        time.sleep(3)

    TimeToSleep()
