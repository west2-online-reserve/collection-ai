import numpy as np
x = np.array([2.0, 1.0, 0.1])
def softmax(x):
    exps=np.exp(x)
    print((exps/np.sum(exps)).shape)
    return exps/np.sum(exps)
print(softmax(x))