import numpy as np
from math import sqrt
points_A = np.empty((5, 2), dtype=int)
points_B = np.empty((8, 2), dtype=int)
for i in range(points_A.shape[0]):
    points_A[i] = np.array(
        [int(np.random.randint(0, 100)), int(np.random.randint(0, 100))])
for i in range(points_B.shape[0]):
    points_B[i] = np.array(
        [int(np.random.randint(0, 100)), int(np.random.randint(0, 100))])
points_A_expanded = points_A[:, np.newaxis, :]
# print(points_A)
# print(points_B)
print(points_A_expanded)
print(points_A_expanded+points_B)
distance_matrix = np.sqrt(np.sum((points_A_expanded - points_B) ** 2, axis=2))
# print(distance_matrix.shape)
min_distances = np.min(distance_matrix, axis=1)
print(min_distances.shape)
boolean_mask = distance_matrix < 20
print(boolean_mask)
ansx = np.where(boolean_mask)[0]
ansy = np.where(boolean_mask)[1]
ans = np.array([ansx, ansy])
print(ans)
