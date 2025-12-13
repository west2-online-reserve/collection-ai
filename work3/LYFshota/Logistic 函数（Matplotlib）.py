import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  

def logistic_function(L, k, x0, x):
    return L / (1 + math.exp(-k * (x - x0)))
# 标准参数设置
L = 1
k = 1
x0 = 0
x_values = np.linspace(-10, 10, 400)
y_values = [logistic_function(L, k, x0, x) for x in
    x_values]
# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=f'Logistic函数: L={L}, k={k}, x0={x0}')
plt.title('Logistic函数曲线')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
# 参数变化对曲线的影响
params = [
    (1, 1, 0),
    (1, 0.5, 0),
    (1, 2, 0),
    (1, 1, -2),
    (1, 1, 2),
    (2, 1, 0),
    (0.5, 1, 0)
]
plt.figure(figsize=(12, 8))
for L, k, x0 in params:
    y_values = [logistic_function(L, k, x0, x) for x in x_values]
    plt.plot(x_values, y_values, label=f'L={L}, k={k}, x0={x0}')
plt.title('不同参数下的Logistic函数曲线')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()

    