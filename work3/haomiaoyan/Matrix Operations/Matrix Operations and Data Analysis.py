import numpy as np

# temp = np.array([[1,2,3,4,5,6],
#                 [5,6,87,1,3,1]])
#
# temp1 = np.array([3,4,5,6,8,9,323])
# print(temp.argmax())  # 输出最大值的索引
# print(temp.argmin())  # 输出最小值的索引
# print(temp.mean())  # 输出平均值
# print(np.median(temp))  # 输出平均值
# print(temp[0, 1:4])  # 输出第二行二到四列
# print(temp[::-1])  # 逆序输出

data_matrix = np.arange(100).reshape([10, 10])
print(data_matrix)

data_matrix_son = data_matrix[3:8, 3:8]  # 3到8行的3到8列----data_matrix_son 类似与切片的用法
print("子矩阵：")
print(data_matrix_son)

data_matrix[data_matrix > 75] = 0  # 判断大小
print('data_matrix:')
print(data_matrix)

data_matrix = data_matrix * 0.8  # 进行与数字的运算
print(data_matrix)

print(data_matrix.max())
max_element_1wei_find = data_matrix.argmax()  # 一维索引
print(max_element_1wei_find)
print(np.unravel_index(max_element_1wei_find, data_matrix.shape))  # 将一维索引转为行列索引

