import numpy as np


# 参数为向量
def my_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([1.0, 2.0, 0.1, 0.2])

outputs = my_softmax(x)

print("输入：", x)
print("输出：", outputs)
