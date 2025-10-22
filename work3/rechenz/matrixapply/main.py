import numpy as np

data_matrix = np.arange(100, dtype=float).reshape(10, 10)
# print(data_matrix)
son_data_matrix = data_matrix[4:8, 4:8]
# print(son_data_matrix)
big_index = np.where(data_matrix > 75)
# print(data_matrix[big_index])
data_matrix[data_matrix > 75] = 0
# print(data_matrix)
data_matrix *= 0.8
print(data_matrix)
max_index = data_matrix.argmax()
max_index = np.unravel_index(max_index, data_matrix.shape)
print(f'行索引:max_index[0]', f'列索引:max_index[1]')
