import numpy as np

def softmax(x):
    # 计算指数
    exp_x = np.exp(x)
    # 计算softmax
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# 输入一个N维向量
x = np.array([1, 2, 3])
print("输入向量形状:", x.shape)

# 调用softmax函数
output = softmax(x)
print("输出向量形状:", output.shape)
print("输出向量:", output)