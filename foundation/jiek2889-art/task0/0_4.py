a = input().split()
x = int(a[0])
y = int(a[1])
num = []
for i in range(x, y + 1):
    if i % 400 == 0 or (i % 4 == 0 and i % 100 != 0):
        num.append(i)
print(len(num))
print(*num)
