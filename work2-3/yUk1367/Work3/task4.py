import numpy as np

points_A = np.random.randint(0, 101, size=(5, 2))
points_B = np.random.randint(0, 101, size=(8, 2))
print(points_A)
print(f"{points_B}\n")

distance_matrix = np.sqrt(np.sum((points_A[:, np.newaxis, :] - points_B)**2, axis=2))
print(distance_matrix)

min_distances = np.min(distance_matrix, axis=1)
print(min_distances)

r = np.where(np.any(distance_matrix < 20, axis=0))[0]
print(r)