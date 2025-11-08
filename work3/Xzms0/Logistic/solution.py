from math import e

import numpy as np
import matplotlib.pyplot as plt


def logistic(L = 1, k = 1, x0 = 0):
    x = np.arange(-10, 10, 0.1, dtype=float)
    y =  L / (1 + e ** (-k * (x - x0)))

    plt.plot(x, y, label = f"L={L}, k={k}, x0={x0}")
    plt.plot(x0, L / 2, marker = 'o')
    plt.text(x0, L / 2, f"({x0}, {L / 2})", ha = "left", va = "top")
    plt.legend()


def main():
    L, k, x0 = 1, 1, 0
    logistic(L = L, k = k, x0 = x0)

    L, k, x0 = 2, 1, 0
    logistic(L = L, k = k, x0 = x0)

    L, k, x0 = 1, 2, 0
    logistic(L = L, k = k, x0 = x0)

    L, k, x0 = 1, 1, -3
    logistic(L = L, k = k, x0 = x0)

    L, k, x0 = 2, 1, -4
    logistic(L = L, k = k, x0 = x0)
    

if __name__ == "__main__":
    main()
    plt.show()