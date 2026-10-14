x, y, z = map(int, input().split())

if x >= y:
    if y >= z:
        print(x, y, z)
    elif x >= z:
        print(x, z, y)
    else:
        print(z, x, y)

else:
    if x >= z:
        print(y, x, z)
    elif y >= z:
        print(y, z, x)
    else:
        print(z, y, x)
# ______________________________________________________
x, y, z = map(int, input().split())

if x < y:
    x, y = y, x

if x < z:
    x, z = z, x

if y < z:
    y, z = z, y

print(x, y, z)
# ______________________________________________________
x, y, z = map(int, input().split())

a = [x, y, z]

a.sort(reverse=True)

print(*a)
# ______________________________________________________
x, y, z = map(int, input().split())

a = [x, y, z]

n = len(a)

for i in range(n - 1):
    for j in range(n - 1 - i):
        if a[j] < a[j + 1]:
            a[j], a[j + 1] = a[j + 1], a[j]

print(*a)
