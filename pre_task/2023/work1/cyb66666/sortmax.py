import numpy as np
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum()
x = np.array([1,2,3,4,5,6,7,8,9]).reshape(9,1)
print(softmax(x))
