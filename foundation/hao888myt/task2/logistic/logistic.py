import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["unifont"]  # 更改字体
    plt.rcParams["axes.unicode_minus"] = False  # 显示负号

    x = np.linspace(-10, 10, 500)

    plt.figure(figsize=(8, 5))
    plt.plot(x, logistic(x, L=1, k=1, x0=0), color="blue", linewidth=2)

    plt.title("标准Logistic函数")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    ax3: Axes = axes[2]

    ax1.plot(x, logistic(x, 1, 1, 0), label="L=1", color="blue")
    ax1.plot(x, logistic(x, 2, 1, 0), label="L=2", color="orange")
    ax1.plot(x, logistic(x, 3, 1, 0), label="L=3", color="green")
    ax1.set_title("改变L")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(x, logistic(x, 1, 0.5, 0), label="k=0.5", color="blue")
    ax2.plot(x, logistic(x, 1, 1, 0), label="k=1", color="orange")
    ax2.plot(x, logistic(x, 1, 3, 0), label="k=3", color="green")
    ax2.set_title("改变k")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3.plot(x, logistic(x, 1, 1, -3), label="x0=-3", color="blue")
    ax3.plot(x, logistic(x, 1, 1, 0), label="x0=0", color="orange")
    ax3.plot(x, logistic(x, 1, 1, 3), label="x0=3", color="green")
    ax3.set_title("改变x0")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
