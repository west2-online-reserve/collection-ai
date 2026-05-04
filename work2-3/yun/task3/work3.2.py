import numpy as np
points_A=np.random.randint(0,101,(5,2))
points_B=np.random.randint(0,101,(8,2))
print(points_A)
print(points_B)
points_A=points_A.reshape(5,1,2)
points_C=points_A-points_B
points_C=points_C**2
distance_matrix=np.sqrt(np.sum(points_C,axis=2))#计算距离矩阵
print(distance_matrix)
print(np.min(distance_matrix,axis=1))
if np.any(distance_matrix<20):
    needed_points = np.unique(np.where(distance_matrix<20)[1])#去重
    print(needed_points)