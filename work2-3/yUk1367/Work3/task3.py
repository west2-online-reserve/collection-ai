import numpy as np
import pandas as pd

data_matrix = np.arange(100).reshape(10, 10)

print(data_matrix[3:7, 3:7])

data_matrix[data_matrix > 75] = 0
print(data_matrix)

data_matrix = data_matrix.astype(float)
data_matrix *= 0.8
print(data_matrix)

print(np.unravel_index(np.argmax(data_matrix), data_matrix.shape))