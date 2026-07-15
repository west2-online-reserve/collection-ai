import matplotlib.pyplot as plt
import numpy as np

def logistic_function(x, L=1, k=1, x0=0):
    return L / (1 + np.exp(-k * (x - x0)))

# 创建图形
plt.figure(figsize=(8, 6))

# 用等差数列生成点
x = np.linspace(-5, 15, 200)  

plt.subplot(2, 2, 1)  # 创建子图
y = logistic_function(x)
plt.plot(x, y, linewidth=1.5, label='L=1, k=1, x₀=0')  # 绘制实线并添加标签
plt.xlabel('x')
plt.ylabel('y')
plt.legend()  # 显示图例

plt.subplot(2, 2, 2)
for argL in [1, 2, 5, 0.5, 0.2]:
    y = logistic_function(x, L=argL, k=1, x0=0)
    plt.plot(x, y, linewidth=1.5, label=f'L={argL}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 3)
for k in [1, 2, 5, 0.5, 0.2]:
    y = logistic_function(x, L=1, k=k, x0=0)
    plt.plot(x, y, linewidth=1.5, label=f'k={k}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 4)
for x0 in [0,2,5,-2,-5]:
    y = logistic_function(x, L=1, k=1, x0=x0)
    plt.plot(x, y, linewidth=1.5, label=f'x₀={x0}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()