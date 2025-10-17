import numpy as np
def softmax(array):
    array=np.exp(array)
    sum=np.sum(array)
    #return array/sum
    print('shape:\n',array.shape,'\nsoftmax:\n',array/sum)
softmax(np.array([[1],[1],[5]]))