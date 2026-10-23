x, y = map(int, input().split())
sum = 0
lis = []
for i in range(x,y+1):
    if i % 100 == 0 and i % 400 == 0:
        sum+=1
        lis.append(i)
    elif i % 4 == 0 and i % 100 != 0:
        sum+=1
        lis.append(i)
print(sum)
print(*lis)
