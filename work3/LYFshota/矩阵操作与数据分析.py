import numpy as np

if __name__ == "__main__":
    data_matrix=np.arange(100).reshape(10,10)
    print("Original Data Matrix:")
    print(data_matrix)
    Submatrix_4x4=data_matrix[2:6,2:6]
    print("Extracted 4x4 Submatrix:")   
    print(Submatrix_4x4)
    data_matrix[data_matrix>75]=0
    data_matrix=data_matrix*0.8
    print(data_matrix)
    ind_max=np.unravel_index(np.argmax(data_matrix,axis=None),data_matrix.shape)
    print(f"Max Value Position: {ind_max}, Max Value: {data_matrix[ind_max]}")