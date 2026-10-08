x, y = map(int, input().split())
n = 0
z = []
for i in range(x, y + 1):
    if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0):
        n += 1
        z.append(i)
print(n)
print(z)
