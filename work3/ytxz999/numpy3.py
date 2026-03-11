import numpy as np
#创建0~99的矩阵
data_matrix = np.arange(0,100,1).reshape(10,10)
#创建最中心的子矩阵
son_matrix = data_matrix[4:6,4:6]
print(son_matrix)
#大于75的设为0
data_matrix[data_matrix > 75] = 0
print(data_matrix)
#每个元素乘0.8
data_matrix=data_matrix * 0.8
print(data_matrix)
#招到最大值
max = data_matrix.max()
print(max)
#找到行列
index = data_matrix.argmax()
row_col = np.unravel_index(index, data_matrix.shape)
print(row_col)