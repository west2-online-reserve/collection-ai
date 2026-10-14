lis = [[1 for _ in range(10)] for _ in range(5)]
print(lis)

def flip_matrix(lis):
    lis = [[row[i] for row in lis] for i in range(10)]
    print(lis)

flip_matrix(lis)