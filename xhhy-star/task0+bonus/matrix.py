matrix = [[1 for col in range(10)] for row in range(5)]

for row in matrix:
    print(row)

transposed = [[matrix[row][col] for row in range(5)] for col in range(10)]
print("\n转置后的矩阵:")
for row in transposed:
    print(row)

#完成矩阵的时候让我想起了九九乘法表（）