import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题


# 定义Logistic函数
def logistic(x, L=1, k=1, x0=0): return L / (1 + np.exp(-k * (x - x0)))

# 生成x轴数据：范围-10到10，取1000个点（点数越多曲线越平滑）
x = np.linspace(-10, 10, 2000)

# 绘制标准Logistic曲线（L=1, k=1, x0=0）
plt.figure(figsize=(10, 8))  # 设置画布大小

# 绘制标准曲线
# matplotlib 的subplot(m, n, p)函数表示：将画布分为m行n列，当前绘制第p个子图（从左到右、从上到下计数）
plt.subplot(2, 2, 1)  # 2行2列的第一个子图
y_standard = logistic(x, L=1, k=1, x0=0)  # 计算y的值

# 绘制曲线  其中‘label’ 需配合plt.legend()显示
plt.plot(x, y_standard, color='red', linewidth=2, label='标准曲线(L=1, k=1, x0=0)')
plt.title('标准Logistic曲线（S形曲线）')  # 标题
plt.xlabel('x')  # x轴的名称
plt.ylabel('f(x)')  # y轴的名称
plt.legend()  # 绘制图例
plt.grid(alpha=0.3)  # 添加网格，alpha设置透明度

# 不同L值的影响（k=1, x0=0固定）
plt.subplot(2, 2, 2)
# 不同L值：0.5、1、2
y_L05 = logistic(x, L=0.5, k=1, x0=0)
y_L1 = logistic(x, L=1, k=1, x0=0)
y_L2 = logistic(x, L=2, k=1, x0=0)

plt.plot(x, y_L05, label='L=0.5', linewidth=2)
plt.plot(x, y_L1, label='L=1', linewidth=2)
plt.plot(x, y_L2, label='L=2', linewidth=2)
plt.title('不同L值对曲线的影响（k=1, x0=0）')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)

# 不同k值的影响（L=1, x0=0固定）
plt.subplot(2, 2, 3)
# 不同k值：0.5（平缓）、1（标准）、2（陡峭）
y_k05 = logistic(x, L=1, k=0.5, x0=0)
y_k1 = logistic(x, L=1, k=1, x0=0)
y_k2 = logistic(x, L=1, k=2, x0=0)

plt.plot(x, y_k05, label='k=0.5（平缓）', linewidth=2)
plt.plot(x, y_k1, label='k=1（标准）', linewidth=2)
plt.plot(x, y_k2, label='k=2（陡峭）', linewidth=2)
plt.title('不同k值对曲线的影响（L=1, x0=0）')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)

# 不同x0值的影响（L=1, k=1固定）
plt.subplot(2, 2, 4)
# 不同x0值：-3（左移）、0（标准）、3（右移）
y_x0_3 = logistic(x, L=1, k=1, x0=-3)
y_x00 = logistic(x, L=1, k=1, x0=0)
y_x03 = logistic(x, L=1, k=1, x0=3)

plt.plot(x, y_x0_3, label='x0=-3（左移）', linewidth=2)
plt.plot(x, y_x00, label='x0=0（标准）', linewidth=2)
plt.plot(x, y_x03, label='x0=3（右移）', linewidth=2)
plt.title('不同x0值对曲线的影响（L=1, k=1）')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)

# 自动调整子图间距，避免标题/标签重叠
plt.tight_layout()
# 显示图像
plt.show()
# 该部分由豆包边教边学完成
