import numpy


def my_softmax(arr):
    lis = [numpy.exp(i) for i in arr]
    sum_num = sum(lis)
    lis = list(map(lambda x: x / sum_num, lis))
    return numpy.array(lis)


print(my_softmax(numpy.array([1, 2, 3, 4, 5])))
# [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
