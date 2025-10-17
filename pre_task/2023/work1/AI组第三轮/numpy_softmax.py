import numpy as np

#实现softmax
def softmax(x):
    # 求e的幂次方
    exp_x = np.exp(x)

    # 归一化
    output = exp_x / np.sum(exp_x, axis=0)

    return output


a1 = np.array([[2.0,5.0],[6.0,3.0]])

output = softmax(a1)

print(output)



