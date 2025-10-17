import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

def logistic(x,L,k,x0):
    return L/(1+np.exp(-k*(x-x0)))

if __name__ == "__main__":
    x=np.linspace(-10,10,2000) #创建一个列表

    plt.figure((figsize(10,6))) #画布

    # L=1, k=1, x0=0
    plt.plot(x, logistic(x, 1, 1, 0), label=r'$L=1, k=1, x_0=0$', linestyle='-', color='b')

    # L=2, k=1, x0=0
    plt.plot(x, logistic(x, 2, 1, 0), label=r'$L=2, k=1, x_0=0$', linestyle='--', color='g')

    # L=1, k=2, x0=0
    plt.plot(x, logistic(x, 1, 2, 0), label=r'$L=1, k=2, x_0=0$', linestyle=':', color='r')

    # L=1, k=1, x0=2
    plt.plot(x, logistic(x, 1, 1, 2), label=r'$L=1, k=1, x_0=2$', linestyle='-.', color='orange')

    # 设置图形的标题和标签
    plt.title("Logistic Function with Varying Parameters")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend() #显示左上角图例

    # 显示图形
    plt.grid(True) #绘制网格线
    plt.show()