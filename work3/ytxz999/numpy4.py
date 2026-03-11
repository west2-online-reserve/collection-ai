import numpy as np

points_A = np.random.randint(0,100,(5,2))
points_B = np.random.randint(0,100,(8,2))
print(points_A)
print(points_B)
#扩展维度，广播
diff = points_A[:,np.newaxis,:] - points_B[np.newaxis,:,:]
distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
print(np.round(distance_matrix,2))
#找最小,设置axis=1
print(np.round(distance_matrix.min(axis=1)))

#找出 points_B 中，与 points_A 中至少一个点的距离小于 20 的所有点的索引。
distance_distin =distance_matrix <20
true_or_false =np.any(distance_distin,axis=0)
indices = np.where(true_or_false)[0]
print(true_or_false)
print(indices)
print(points_B[indices])


