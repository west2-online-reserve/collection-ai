import numpy as np


def softmax(x):

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素


    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


x = np.random.randint(low=1, high=5, size=(2, 3))  # 生成一个2x3的矩阵，取值范围在1-5之间
print("原始 ：\n", x)
x_ = softmax(x)
print("变换后 ：\n", x_)