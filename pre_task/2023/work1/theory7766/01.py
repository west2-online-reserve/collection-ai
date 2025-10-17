import numpy as np
def softmax(x):
    # 减去最大值消除最大
    x -= np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x/np.sum(exp_x)
    return softmax_x

def main():
    a = np.linspace(10, 100, 10)
    print(a, a.shape)
    b = softmax(a)
    print(b, b.shape)

    # 一个比较大的数
    c = np.array([100000., 1., 100.])
    print(c, c.shape)
    d = softmax(c)
    print(d, d.shape)

if __name__=="__main__":
    main()
