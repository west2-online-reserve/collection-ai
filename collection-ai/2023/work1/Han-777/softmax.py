import numpy as np
import time
import datetime

"""
1. softmax
缺点；可能发生溢出
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
"""

"""
2. 差值softmax
先计算最大值，在对元素与最大值的差值矩阵进行softmax处理
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
"""

"""
3. log_softmax
softmax将数值压缩至0-1；log_softmax将数据压缩至负无穷到0
优点：对数可将除法转化为减法运算从而节省时间
torch.nn.CrossEntropyLoss == softmax + log + nllloss
def log_softmax(x):
    mx = np.amax(x, axis=0, keepdims=True)
    return np.log(np.exp(x - mx) / np.sum(np.exp(x - mx), axis=0, keepdims=True))
"""

__doc__ = """Input the vector and print the softmax result.
>>> 1 2 3 4 5
softmax: [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
softmax execute time: 0.0005005s
log_softmax: [-4.4519144 -3.4519144 -2.4519144 -1.4519144 -0.4519144]
log_softmax execute time: 0.0000457s
>>> [1, 2, 3, 4, 5]
softmax: [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
softmax execute time: 0.0001020s
log_softmax: [-4.4519144 -3.4519144 -2.4519144 -1.4519144 -0.4519144]
log_softmax execute time: 0.0000327s
"""


def func_execute_time(iteration):
    def inner(func):
        def wrapper(*args, **kwargs):
            global result
            start = time.perf_counter()
            for i in range(iteration):
                result = func(*args, **kwargs)
            print(f"{func.__name__} execute time: {time.perf_counter() - start:.7f}s")
            return result

        return wrapper

    return inner


@func_execute_time(1)
def softmax(x):
    # axis=0时列比较, axis=1时行比较, keepdims=True,保持维度不变
    mx = np.amax(x, axis=0, keepdims=True)
    return np.exp(x - mx) / np.sum(np.exp(x - mx), axis=0, keepdims=True)


@func_execute_time(1)
def log_softmax(x):
    mx = np.amax(x, axis=0, keepdims=True)
    return np.log(np.exp(x - mx) / np.sum(np.exp(x - mx), axis=0, keepdims=True))


def main():
    input_inf = input("input your Matrix:")
    try:
        input_matrix = eval(input_inf)
    except:
        input_matrix = list(map(eval, input_inf.split(" ")))

    print(f"softmax: {softmax(input_matrix)}")
    print(f"log_softmax: {log_softmax(input_matrix)}")


print(__doc__)

while __name__ == '__main__':
    main()
