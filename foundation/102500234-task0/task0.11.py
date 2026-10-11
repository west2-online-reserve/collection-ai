
matrix = [[1 for _ in range(10)] for _ in range(5)]
for r in matrix:
    print(r)


transposed = [[matrix[i][j] for i in range(5)] for j in range(10)]
for r in transposed:
    print(r)
