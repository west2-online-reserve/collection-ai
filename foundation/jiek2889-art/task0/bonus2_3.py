x = 1
a = [[x for i in range(10)] for j in range(5)]
print("\n".join([" ".join([str(item) for item in row]) for row in a]))
b = [[row[i] for row in a] for i in range(10)]
print("\n".join([" ".join([str(item) for item in row]) for row in b]))
