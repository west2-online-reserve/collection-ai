import numpy as np
import matplotlib.pyplot as plt
from math import e


def main():
    x = np.arange(-10, 10, 0.1, dtype=float)
    L = 1
    k = 1
    x0 = 1
    y = calc(x, L, k, x0)
    plt.plot(x, y)
    plt.plot(x0, L/2, marker='o')
    L = 2
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='r')
    L = 3
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='r')
    L = 1
    k = 2
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='g')
    k = 3
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='g')
    k = 1
    x0 = 2
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='b')
    x0 = 3
    y = calc(x, L, k, x0)
    plt.plot(x, y, color='b')
    plt.show()


def calc(x, L, k, x0):
    return L/(1+e**(-k*(x-x0)))


if __name__ == '__main__':
    main()
