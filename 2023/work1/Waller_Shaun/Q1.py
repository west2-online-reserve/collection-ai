#用numpy实现softmax函数，输入一个N维度向量输入一个N维向量，numpy数组可以调用.shape查看形状，输入输出的形状都为(n,)。
#softmax(x)是一种激活函数，用于将一组数值转换为概率分布，通常部署在神经网络的输出层
#指数化使他所有输出值都是正数，标准化使之形成概率分布
import numpy as np
def softmax(x):
    e_x = np.exp(x-max(x)) #减去最大值，防止指数函数溢出，并能保持计算结果不变（减去相同的数，且计算的是相对大小）
    return e_x/(sum(e_x))
x = np.array([1,2,3,4,5,6])
x_out = softmax(x)
print('输入形状：{}'.format(x.shape))
print('输出形状：{}'.format(x_out.shape))
print(x_out)
