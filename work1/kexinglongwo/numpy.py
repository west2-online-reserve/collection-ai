import numpy as np
def softmax(x):#e**j/(和e**j)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
