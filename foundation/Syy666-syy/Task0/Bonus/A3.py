m=[[1 for _ in range(10)]for _ in range(5)]
T=[[m[i][j] for i in range(5)] for j in range(10)]
print("原矩阵:")
for row in m:
    print(row)
print("\n转置矩阵：")
for row in T:
    print(row)
