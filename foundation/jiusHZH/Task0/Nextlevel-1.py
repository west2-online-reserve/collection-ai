import random
import time
def outer(func):
    def inner():
        print(f"Function name: {func.__name__}")
        start_time = time.time()
        func()
        end_time = time.time()
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Function took {end_time - start_time} seconds to execute")
    return inner
@outer
def dummy_function():
    a=[1.23,43,21,222,334,"1",31,"22222","11www"]
    for i in range(100):
        random.shuffle(a)
    return
if __name__ == "__main__":
    dummy_function()