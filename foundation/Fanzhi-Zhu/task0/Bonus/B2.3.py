matrix = []

for i in range(5):          
    row = []                
    for j in range(10):     
        row.append(1)       
    matrix.append(row)      

print("原矩阵（5 行 10 列）：")
for row in matrix:
    print(row)

transposed = []

for j in range(10):         
    new_row = []            
    for i in range(5):      
        new_row.append(matrix[i][j])  
    transposed.append(new_row)

print("\n转置矩阵（10 行 5 列）：")
for row in transposed:
    print(row)