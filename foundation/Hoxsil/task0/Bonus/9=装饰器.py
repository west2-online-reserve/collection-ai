import time, functools


def metric(fn):
    @functools.wraps(fn)
    def wrapper(*arg, **wkargs):
        start_time = time.time()
        print(f"name:{fn.__name__}")
        x = fn(*arg, **wkargs)
        end_time = time.time()
        print(f'用时 {(end_time - start_time)*1000} ms')
        return x
    return wrapper


@metric
def f(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

# 测试
g = f
s = g(11, 22, 33)
print(s)
