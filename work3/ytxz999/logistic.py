import matplotlib.pyplot as plt
import numpy as np

#解决汉字无法显示
from matplotlib import rcParams #字体
rcParams['font.family'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

#函数定义
def logistic(x):
    L=1
    k=1
    x0=0
    return L/(1+np.exp(-k*(x-x0)))

#创建图表,设置大小
plt.figure(figsize=(10,5))

#随机取200个点
x = np.linspace(-10,10,200)
y =logistic(x)

plt.plot(x,y,label="logistic",linewidth=3)
#设置标题
plt.title("logistic",fontsize=20)
plt.xlabel("x",fontsize=20,loc="right")
plt.ylabel("y",fontsize=20,loc="top",rotation=0)
#设置y上的步长
plt.yticks(np.arange(0, 1, 0.1))
#添加网格线
plt.grid(True ,alpha = 0.3,color='black',linestyle='--')
plt.show()


