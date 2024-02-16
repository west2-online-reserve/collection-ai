import numpy

__doc__ = """Input the vector and print the softmax result.

>>> 1 2 3 4 5
[0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
>>> [1, 2, 3, 4, 5]
[0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
"""

def softmax(x):  
    # 对输入向量进行指数化处理，减去最大值以防止数值溢出
    exp_x = numpy.exp(x - numpy.max(x))
    # 返回归一化后的概率分布
    return exp_x / exp_x.sum()

def main():
    get = input(">>>")
    try:
        data = eval(get)
    except:
        data = list( map(eval, get.split(" ")) )
    
    vector = numpy.array(data)
    result = softmax(vector)
    print(result)

print(__doc__)

while(__name__ == '__main__'):
    main()