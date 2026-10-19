a = []
for i in range(1, 10):
    b = []
    for j in range(1, i + 1):
        b.append(f"{i}*{j}={i*j}")
    a.append(b)
print(a)
