import numpy as np
if __name__ == "__main__":
    points_A=np.random.randint(0,100,size=(5,2))
    points_B=np.random.randint(0,100,size=(8,2))
    points_A_expanded=points_A[:,np.newaxis,:]
    points_B_expanded=points_B[np.newaxis,:,:]
    distance_matrix=np.array(np.sqrt(np.sum((points_A_expanded - points_B_expanded)**2,axis=-1)))
    print("Distance Matrix between points in A and B:")
    print(distance_matrix)
    min_distance= np.min(distance_matrix,axis=1)
    print(min_distance)
    mask_lt_20 = distance_matrix < 20
    b_has_close_a = np.any(mask_lt_20, axis=0)
    b_indices = np.where(b_has_close_a)[0]
    print("Indices in points_B with distance < 20 to any point in points_A:")
    print(b_indices)