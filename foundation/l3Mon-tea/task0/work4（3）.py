a =int(input())
b =int(input())
y = []
T = 0
for i in range(a,b + 1):
    if (i %4 == 0 and i %100 != 0) or i %400 == 0:
        T += 1
        y.append(i)
print(T)
print(*y)