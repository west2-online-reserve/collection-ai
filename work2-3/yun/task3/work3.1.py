import numpy as np
data_matrix= np.arange(0,100,1).reshape(10,10)
print(data_matrix)
fourfour=data_matrix[data_matrix.shape[0]//2-1:data_matrix.shape[0]//2+1,data_matrix.shape[1]//2-1:data_matrix.shape[1]//2+1]
print(fourfour)
need=np.where(data_matrix>75)
data_matrix[(data_matrix>75)]=0
print(data_matrix)
data_matrix=data_matrix*0.8
print(data_matrix)
print(np.max(data_matrix))
print(np.unravel_index(np.argmax(data_matrix), data_matrix.shape))