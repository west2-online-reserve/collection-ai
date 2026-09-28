a = input().split()
b = int(a[0])
c = int(a[1])
count = 0
runnian = []
for i in range(b,c+1):
    if (i % 4==0 and i % 100 != 0) or i % 400 == 0:
        count += 1
        runnian.append(i)
print(count)
for j in runnian:
    print(j, end=' ')


