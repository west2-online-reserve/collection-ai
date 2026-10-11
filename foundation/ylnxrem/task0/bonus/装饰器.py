import time, functools


def log(fn):
    @functools.wraps(fn)
    def logger(*s, **g):
        print("begin", f"{fn.__name__}")
        start = time.time()
        r = fn(*s, **g)
        endt = time.time()
        print(r, "开始时间：", start, "结束时间：", endt, "执行时间：", endt - start)
        return r

    return logger