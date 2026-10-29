a = input().split()
idx = 0
b = []
for i in range(int(a[0]),int(a[1])+1):
    if (i % 4 == 0 and i % 100 != 0) or i % 400 ==0:
        idx += 1
        b.append(i)
print(idx)
for i in range(0,idx):
    print(f"{b[i]}",end=" ")
