# 写一个列表推导式
# 生成一个 5*10 的矩阵,矩阵内的所有值为 1
# 再写一个列表推导式，把这个矩阵转置

matrix = [[1 for i in range(10)] for i in range(5)]
print(matrix)

transpose = [[matrix[i][j] for i in range(5)] for j in range(5)]
print(transpose)